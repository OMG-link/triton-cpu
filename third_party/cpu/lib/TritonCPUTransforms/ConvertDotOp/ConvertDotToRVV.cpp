#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTORVV
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

enum DotStyle { OUTER, INNER };

// This structure is used to hold candidates for conversion to RVV operations.
struct RvvDotOpCandidate {
  // Operation to convert.
  cpu::DotOp op;
  // Type of input and output element.
  Type inputElemTy;
  Type outputElemTy;

  // Matrix sizes. LHS = <m x k>; RHS = <k x n>; output = <m x n>
  int64_t m, k, n;

  // The way we do GEMM. Support: outer-product(OUTER), inner-product(INNER)
  DotStyle dotStyle;

  // TransferReadOps that read input matrix.
  // If any operand of dot is not a TransferReadOp, it will be stored to a
  // temporary buffer and read out using TransferReadOp.
  vector::TransferReadOp lhsReadOp, rhsReadOp;
};

// Check if input types are same, and if output elemets types are same with
// input or double-width relative to input. If success, inputElemTy and
// outputElemTy in candidate are filled.
bool checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                    Type resElemTy, RvvDotOpCandidate &candidate) {
  MLIRContext *ctx = lhsElemTy.getContext();

  // inputElemTy
  Type inputElemTy = lhsElemTy;
  if (rhsElemTy != inputElemTy) {
    LDBG("checkElemTypes failed: rhsElem has type "
         << rhsElemTy << ", which differs from lhsElem's type " << lhsElemTy);
    return false;
  }
  if (!inputElemTy.isIntOrFloat()) {
    LDBG("checkElemTypes failed: inputElemTy is "
         << inputElemTy << ", which is neither integer nor float");
    return false;
  }
  candidate.inputElemTy = inputElemTy;

  // outputElemTy
  Type outputElemTy = resElemTy;
  if (accElemTy != outputElemTy) {
    LDBG("checkElemTypes failed: accElem has type "
         << accElemTy << ", which differs from resElem's type " << resElemTy);
    return false;
  }
  if (!outputElemTy.isIntOrFloat()) {
    LDBG("checkElemTypes failed: outputElemTy is "
         << outputElemTy << ", which is neither integer nor float");
    return false;
  }
  if (inputElemTy.isInteger() != outputElemTy.isInteger()) {
    LDBG("checkElemTypes failed: inputElemTy and outputElemTy must be both int "
         "or float. Currently, they are (input)"
         << inputElemTy << " and (output)" << outputElemTy);
    return false;
  }
  candidate.outputElemTy = outputElemTy;

  return true;
}

// Check input shapes. Currently, support only 2D cases and ignore small
// inputs. Sizes field of candidate is filled if check passed.
bool checkInputShapes(VectorType lhsTy, VectorType resTy,
                      RvvDotOpCandidate &candidate) {
  if (lhsTy.getRank() != 2) {
    LDBG(
        "checkInputShapes failed: lhsTy has invalid rank: " << lhsTy.getRank());
    return false;
  }

  // Fillin matrix size
  candidate.m = resTy.getDimSize(0);
  candidate.n = resTy.getDimSize(1);
  candidate.k = lhsTy.getDimSize(1);

  return true;
}

vector::TransferReadOp findInputTransferRead(Value inputMat) {
  if (auto transferReadOp =
          dyn_cast<vector::TransferReadOp>(inputMat.getDefiningOp())) {
    // FIXME: We should check if data over tramsfer_read.base may be overwriiten
    // between transfer_read and dot. If so, this transfer_read is invalid.
    return transferReadOp;
  }
  // Return an empty op which will be replaced later.
  // We can't return an op that reads from temporary buffer because we don't
  // have a rewriter here.
  return vector::TransferReadOp();
}

// Returns the first dimension with stride 1, which is usually considered as
// 'lowest' dimention.
// The result would be -1 if no dimension with stride 1 were found.
int64_t findLowestDim(MemRefType memRefType) {
  llvm::SmallVector<int64_t, 8> strides;
  int64_t offset = 0;
  if (succeeded(memRefType.getStridesAndOffset(strides, offset))) {
    // Find the dimension with stride 1
    int64_t lowestDim = -1;
    for (size_t i = 0; i < strides.size(); ++i) {
      int64_t s = strides[i];
      if (s == 1) {
        lowestDim = static_cast<int64_t>(i);
        break;
      }
    }
    return lowestDim;
  } else {
    return -1;
  }
}

/**
 * Finds the dimension of the result of TransferReadOp that was read
 * contiguously from memory.
 *
 * returns:
 *  - -1 if the result is unknown
 *  - any non-negative number indicating that the target dimension is the
 *    result-th FROM THE END (0-based)
 */
int64_t findLowestDim(vector::TransferReadOp transferReadOp) {
  if (!transferReadOp) {
    // storeToTempBuffer will make the last dimension continuous.
    return 0;
  }

  auto memRefType = dyn_cast<MemRefType>(transferReadOp.getBase().getType());
  if (!memRefType) {
    // Don't know how to find lowest dimension if memref is not memref.
    // Return -1 that represents unknown.
    return -1;
  }

  // Firstly find the lowest dimension of memref
  int64_t memLowestDim = findLowestDim(memRefType);
  if (memLowestDim == -1) {
    return -1;
  }

  // Then find a dimension that was permuted to memLowestDim
  auto permMap = transferReadOp.getPermutationMap();
  auto memLowestDimExpr =
      mlir::getAffineDimExpr(memLowestDim, permMap.getContext());
  auto vecLowestDim = permMap.getResultPosition(memLowestDimExpr);

  if (vecLowestDim.has_value()) {
    return transferReadOp.getType().getRank() - vecLowestDim.value() - 1;
  } else {
    return -1;
  }
}

// Determine dot style by input data layout.
void determineDotStyle(Value a, Value b, RvvDotOpCandidate &candidate) {
  candidate.lhsReadOp = findInputTransferRead(a);
  candidate.rhsReadOp = findInputTransferRead(b);
  int64_t lhsLowestDim = findLowestDim(candidate.lhsReadOp);
  int64_t rhsLowestDim = findLowestDim(candidate.rhsReadOp);
  if (rhsLowestDim == 0) {
    // When the last dimension of right operand(N) is continuous, we use
    // outer-product GEMM.
    LDBG("Last dimension of right operand is continuous. "
         "Recommend outer-product GEMM.");
    candidate.dotStyle = OUTER;
  } else if (rhsLowestDim == 1 && lhsLowestDim == 0) {
    // When the penultimate dimension of right operand(K) and the last dimension
    // of left operand(K) is continuous, we use inner-product GEMM.
    LDBG("Penultimate dimension of right operand is continuous and "
         "last dimension of left operand is continuous. "
         "Recommend inner-product GEMM.");
    candidate.dotStyle = INNER;
  } else {
    LDBG("No recommend rule matches. Recommend outer-product GEMM as default "
         "dot "
         "style. (lhsLowestDim="
         << lhsLowestDim << ", rhsLowestDim=" << rhsLowestDim << ")");
    candidate.dotStyle = OUTER;
  }
}

// Check if specified ContractionOp can be lowered to RVV operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isRvvCandidate(cpu::DotOp op, RvvDotOpCandidate &candidate) {
  MLIRContext *ctx = op.getContext();
  VectorType lhsTy = op.getA().getType();
  VectorType rhsTy = op.getB().getType();
  VectorType accTy = op.getC().getType();
  VectorType resTy = op.getType();

  LDBG("Considering candidate op: " << op);
  // Check if input and output types match available hardware capabilities.
  // If check is successful then effective element types are assigned to the
  // candidate.
  if (!checkElemTypes(lhsTy.getElementType(), rhsTy.getElementType(),
                      accTy.getElementType(), resTy.getElementType(),
                      candidate))
    return false;

  // Check input shapes.
  // If check is successful then matrix sizes are assigned to the candidate.
  if (!checkInputShapes(lhsTy, resTy, candidate))
    return false;

  candidate.op = op;

  determineDotStyle(op.getA(), op.getB(), candidate);

  return true;
}

/**
 * Load a sub matrix of the given main matrix.
 *
 * @param mainReadOp: the given main matrix
 * @param subShape: shape of sub matrix
 * @param subIndices: offsets relative to the main matrix
 */
Value loadSubMat(PatternRewriter &rewriter, Location loc,
                 vector::TransferReadOp mainReadOp, VectorType subMatTy,
                 ArrayRef<Value> subIndices, ArrayRef<bool> subInBounds) {
  assert(subIndices.size() == mainReadOp.getType().getRank());
  assert(subInBounds.size() == mainReadOp.getType().getRank());

  auto permMap = mainReadOp.getPermutationMap();

  // Add offset relative to the main matrix
  SmallVector<Value> mainOffIndices(mainReadOp.getIndices());
  for (int i = 0; i < subIndices.size(); i++) {
    auto expr = permMap.getResult(i);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int64_t memDim = dimExpr.getPosition();
      mainOffIndices[memDim] = arith::AddIOp::create(
          rewriter, loc, mainOffIndices[memDim], subIndices[i]);
    } else if (auto constantExpr = dyn_cast<AffineConstantExpr>(expr)) {
      int64_t constant = constantExpr.getValue();
      assert(constant == 0);
      assert(subMatTy.getDimSize(i) == 1);
    } else {
      assert(false && "Don't know how to process this permutation map");
    }
  }

  // Apply inbounds attribute
  auto mainOffInBounds = mainReadOp.getInBoundsValues();
  for (int i = 0; i < subInBounds.size(); i++) {
    mainOffInBounds[i] = (mainOffInBounds[i] & subInBounds[i]);
  }

  // Create new TransferReadOp
  auto mainOffInBoundsAttr = rewriter.getBoolArrayAttr(mainOffInBounds);
  auto subReadOp = vector::TransferReadOp::create(
      rewriter, loc, subMatTy, mainReadOp.getBase(), mainOffIndices, permMap,
      mainReadOp.getPadding(), mainReadOp.getMask(), mainOffInBoundsAttr);

  return subReadOp;
}

// Load mat[m, n] directly from memory.
Value loadScalar(PatternRewriter &rewriter, Location loc,
                 vector::TransferReadOp matReadOp, Value m, Value n) {
  auto subMatTy = VectorType::get(ArrayRef<int64_t>({1, 1}),
                                  matReadOp.getType().getElementType());
  auto mat = loadSubMat(rewriter, loc, matReadOp, subMatTy,
                        ArrayRef<Value>{m, n}, ArrayRef<bool>{true, true});
  auto scalar =
      vector::ExtractOp::create(rewriter, loc, mat, ArrayRef<int64_t>({0, 0}));
  return scalar;
}

Value loadRow(PatternRewriter &rewriter, Location loc,
              vector::TransferReadOp matReadOp, Value m, Value n,
              VectorType resTy, bool inBounds) {
  assert(resTy.getRank() == 1);
  assert(resTy.getElementType() == matReadOp.getType().getElementType());
  auto subMatTy = VectorType::get(ArrayRef<int64_t>({1, resTy.getDimSize(0)}),
                                  matReadOp.getType().getElementType(),
                                  ArrayRef<bool>({false, resTy.isScalable()}));
  auto mat =
      loadSubMat(rewriter, loc, matReadOp, subMatTy, ArrayRef<Value>({m, n}),
                 ArrayRef<bool>({true, inBounds}));
  auto vec = vector::ShapeCastOp::create(rewriter, loc, resTy, mat);
  return vec;
}

Value loadCol(PatternRewriter &rewriter, Location loc,
              vector::TransferReadOp matReadOp, Value m, Value n,
              VectorType resTy, bool inBounds) {
  assert(resTy.getRank() == 1);
  assert(resTy.getElementType() == matReadOp.getType().getElementType());
  auto subMatTy = VectorType::get(ArrayRef<int64_t>({resTy.getDimSize(0), 1}),
                                  matReadOp.getType().getElementType(),
                                  ArrayRef<bool>({resTy.isScalable(), false}));
  auto mat =
      loadSubMat(rewriter, loc, matReadOp, subMatTy, ArrayRef<Value>({m, n}),
                 ArrayRef<bool>({inBounds, true}));
  auto vec = vector::ShapeCastOp::create(rewriter, loc, resTy, mat);
  return vec;
}

/**
 * Get VMUL according to the number of accumulating vectors.
 */
int64_t getVmul(int64_t numAcc, int64_t accBits) {
  // Assume VMUL=1, we need n VREGs for accumulate and 2 VREGs for input (with
  // prefetch buffer).
  int64_t vmul = 32 / (numAcc + 2);
  // Floor to a power of 2 since:
  // - VMUL must be a power of 2.
  // - We will use more than 32 VREGs if we ceil it.
  vmul = 1ll << (63 - __builtin_clzll(vmul));
  // Prevent a situation where more than half of the registers are left unused.
  if (auto vlen = rvv::getVlen(); vlen > 0) {
    int64_t vregNeeded = (accBits + vlen - 1) / vlen;
    // Ceil to a power of 2
    int64_t maxVmul = 1ll << (64 - __builtin_clzll(vregNeeded - 1));
    vmul = std::min<int64_t>(vmul, maxVmul);
  }
  // VMUL must be at most 8. (For RVV 1.0)
  // The lower bound is set to 1 to maximize register utilization.
  vmul = std::max<int64_t>(vmul, 1);
  vmul = std::min<int64_t>(vmul, 8);
  return vmul;
}

/*
 * Cast 'val' to have element type 'dstElemTy' if needed.
 * If 'val' is a vector, each element is casted.
 */
Value maybeCast(Location loc, Value val, Type dstElemTy,
                PatternRewriter &rewriter) {
  Type srcTy = val.getType();
  Type srcElemTy;
  if (auto srcVecTy = dyn_cast<VectorType>(srcTy); srcVecTy) {
    srcElemTy = srcVecTy.getElementType();
  } else {
    srcElemTy = srcTy;
  }
  if (srcElemTy == dstElemTy)
    return val;
  Type dstTy;
  if (auto srcVecTy = dyn_cast<VectorType>(srcTy); srcVecTy) {
    dstTy = srcVecTy.cloneWith(std::nullopt, dstElemTy);
  } else {
    dstTy = dstElemTy;
  }

  if (!srcElemTy.isIntOrFloat() || !dstElemTy.isIntOrFloat()) {
    llvm_unreachable("maybeCast supports only int or float types.");
  }

  if (srcElemTy.isInteger()) {
    if (srcElemTy.getIntOrFloatBitWidth() < dstElemTy.getIntOrFloatBitWidth())
      return arith::ExtSIOp::create(rewriter, loc, dstTy, val);
    return arith::TruncIOp::create(rewriter, loc, dstTy, val);
  } else {
    if (srcElemTy.getIntOrFloatBitWidth() < dstElemTy.getIntOrFloatBitWidth())
      return arith::ExtFOp::create(rewriter, loc, dstTy, val);
    return arith::TruncFOp::create(rewriter, loc, dstTy, val);
  }
}

LogicalResult convertToOuterProductGemm(RvvDotOpCandidate &candidate,
                                        PatternRewriter &rewriter,
                                        MemBuffer accBuf, bool isAccZeroInit) {
  cpu::DotOp dotOp = candidate.op;
  vector::TransferReadOp lhsReadOp = candidate.lhsReadOp;
  vector::TransferReadOp rhsReadOp = candidate.rhsReadOp;
  Type inputElemTy = candidate.inputElemTy;
  Type outputElemTy = candidate.outputElemTy;
  int64_t mat_m = candidate.m;
  int64_t mat_n = candidate.n;
  int64_t mat_k = candidate.k;

  Location loc = dotOp.getLoc();

  int64_t outputElemBitWidth = outputElemTy.getIntOrFloatBitWidth();
  // If mat_k==1, the result is available immediately. No accumulation is
  // needed.
  int64_t vmul = getVmul(/*numAcc=*/mat_k > 1 ? mat_m : 1,
                         /*accBits=*/mat_n * outputElemBitWidth);
  const int64_t baseVlen = 64;
  const int64_t baseVlmax = vmul * baseVlen / outputElemBitWidth;

  Value baseVlmax_cIndex = index_cst(baseVlmax);

  Value vscale = rvv::getVscale(rewriter, loc);
  Value vlmax = op_muli(vscale, baseVlmax_cIndex);

  VectorType outputMatTy = cast<VectorType>(dotOp.getC().getType());
  VectorType inputSubVecTy, outputSubVecTy;
  VectorType inputSubMatTy, outputSubMatTy;
  if (auto vscaleDefOp = vscale.getDefiningOp<arith::ConstantIndexOp>()) {
    inputSubVecTy = VectorType::get({baseVlmax * vscaleDefOp.value()},
                                    inputElemTy, {false});
    outputSubVecTy = VectorType::get({baseVlmax * vscaleDefOp.value()},
                                     outputElemTy, {false});
    inputSubMatTy = VectorType::get({mat_m, baseVlmax * vscaleDefOp.value()},
                                    inputElemTy, {false, false});
    outputSubMatTy = VectorType::get({mat_m, baseVlmax * vscaleDefOp.value()},
                                     outputElemTy, {false, false});
  } else {
    inputSubVecTy = VectorType::get({baseVlmax}, inputElemTy, {true});
    outputSubVecTy = VectorType::get({baseVlmax}, outputElemTy, {true});
    inputSubMatTy =
        VectorType::get({mat_m, baseVlmax}, inputElemTy, {false, true});
    outputSubMatTy =
        VectorType::get({mat_m, baseVlmax}, outputElemTy, {false, true});
  }

  // We will do GEMM by multiplying an <M x 1> vector with an <1 x N> vector.
  // To do so, we multiply <1 x 1> scalar with <1 x N> vector.
  // As N is given by the user, it can be too large for single vector register.
  // We need to divide <1 x N> vector into <1 x VLMAX> one, where VLMAX is the
  // number of elements single vector register can hold.

  // Code generator that computes <M x K> x <K x VLMAX>.
  auto genForBodyN = [&](Value iv, bool inBounds) {
    Value subVecOff = op_muli(iv, vlmax);

    SmallVector<Value> accIndices = accBuf.indices;
    accIndices[1] =
        arith::AddIOp::create(rewriter, loc, accIndices[1], subVecOff);

    Value accMat;
    if (isAccZeroInit) {
      accMat = ub::PoisonOp::create(rewriter, loc, outputSubMatTy);
    } else {
      accMat = vector::TransferReadOp::create(
          rewriter, loc, outputSubMatTy, accBuf.memRef, accIndices,
          std::nullopt, ArrayRef<bool>({true, inBounds}));
    }

    auto doMul = [&](Value lhsScalar, Value rhsVec) -> Value {
      if (lhsScalar.getType().isIntOrFloat()) {
        lhsScalar = maybeCast(loc, lhsScalar, outputElemTy, rewriter);
        rhsVec = maybeCast(loc, rhsVec, outputElemTy, rewriter);
        auto splat = createBroadcast(
            rewriter, loc, cast<VectorType>(rhsVec.getType()), lhsScalar);
        if (lhsScalar.getType().isInteger()) {
          return arith::MulIOp::create(rewriter, loc, splat, rhsVec);
        } else {
          return arith::MulFOp::create(rewriter, loc, splat, rhsVec,
                                       arith::FastMathFlags::none);
        }
      } else {
        // report type of lhsScalar is unexpected
        llvm_unreachable("Unexpected type of lhsScalar in doMul.");
      }
    };

    auto doMacc = [&](Value accVec, Value lhsScalar, Value rhsVec) -> Value {
      if (lhsScalar.getType().isIntOrFloat()) {
        lhsScalar = maybeCast(loc, lhsScalar, outputElemTy, rewriter);
        rhsVec = maybeCast(loc, rhsVec, outputElemTy, rewriter);
        assert(accVec.getType() == rhsVec.getType());
        auto splat = createBroadcast(
            rewriter, loc, cast<VectorType>(rhsVec.getType()), lhsScalar);
        if (lhsScalar.getType().isInteger()) {
          auto mul = arith::MulIOp::create(rewriter, loc, splat, rhsVec);
          return arith::AddIOp::create(rewriter, loc, accVec, mul);
        } else {
          return vector::FMAOp::create(rewriter, loc, splat, rhsVec, accVec);
        }
      } else {
        // report type of lhsScalar is unexpected
        llvm_unreachable("Unexpected type of lhsScalar in doMacc.");
      }
    };

    for (int64_t k = 0; k < mat_k; k++) {
      Value cIndex_k = index_cst(k);
      Value rhsVec = loadRow(rewriter, loc, rhsReadOp, cIndex_k, subVecOff,
                             inputSubVecTy, inBounds);
      for (int64_t m = 0; m < mat_m; ++m) {
        Value lhsScalar =
            loadScalar(rewriter, loc, lhsReadOp, index_cst(m), cIndex_k);
        Value newAccVec;
        if (isAccZeroInit && k == 0) {
          newAccVec = doMul(lhsScalar, rhsVec);
        } else {
          Value oldAccVec = vector::ExtractOp::create(rewriter, loc, accMat, m);
          newAccVec = doMacc(oldAccVec, lhsScalar, rhsVec);
        }
        accMat = vector::InsertOp::create(rewriter, loc, newAccVec, accMat, m);
      }
    }
    vector::TransferWriteOp::create(rewriter, loc, accMat, accBuf.memRef,
                                    accIndices,
                                    ArrayRef<bool>({true, inBounds}));
  };

  // Divide <K x N> vector into <K x VLMAX> ones.
  // We do this first to achieve the best performance.
  Value numSubVec =
      arith::DivSIOp::create(rewriter, loc, index_cst(mat_n), vlmax);
  auto forOpN =
      scf::ForOp::create(rewriter, loc, index_cst(0), numSubVec, index_cst(1));
  // Process each <M x K> x <K x VLMAX> sub-matrix-product.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOpN.getBody());
    genForBodyN(forOpN.getInductionVar(), true);
  }
  // Process the remaining <K x (N % VLMAX)> vector if N is not a multiple of
  // VLMAX.
  Value nModVl = arith::RemSIOp::create(rewriter, loc, index_cst(mat_n), vlmax);
  auto ifOp = scf::IfOp::create(rewriter, loc,
                                arith::CmpIOp::create(rewriter, loc,
                                                      arith::CmpIPredicate::ne,
                                                      nModVl, index_cst(0)),
                                /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.getBody());
    genForBodyN(numSubVec, false);
  }

  // The result is in accBuf. We should load it and replace the original
  // constraction result.
  Value newAccMat = op_read(outputMatTy, accBuf.memRef, accBuf.indices);
  rewriter.replaceOp(dotOp, newAccMat);

  return success();
}

LogicalResult convertToInnerProductGemm(RvvDotOpCandidate &candidate,
                                        PatternRewriter &rewriter,
                                        MemBuffer accBuf, bool isAccZeroInit) {
  cpu::DotOp dotOp = candidate.op;
  vector::TransferReadOp lhsReadOp = candidate.lhsReadOp;
  vector::TransferReadOp rhsReadOp = candidate.rhsReadOp;
  Type inputElemTy = candidate.inputElemTy;
  Type outputElemTy = candidate.outputElemTy;
  int64_t mat_m = candidate.m;
  int64_t mat_n = candidate.n;
  int64_t mat_k = candidate.k;

  Location loc = dotOp.getLoc();

  int64_t inputElemBitWidth = inputElemTy.getIntOrFloatBitWidth();
  int64_t outputElemBitWidth = outputElemTy.getIntOrFloatBitWidth();
  const int64_t baseVlen_i64 = 64;
  const int64_t baseVlmax_i64 = baseVlen_i64 / outputElemBitWidth;
  assert(baseVlmax_i64 > 0);

  Value baseVlmax_cIndex = index_cst(baseVlmax_i64);

  Value vscale = rvv::getVscale(rewriter, loc);
  Value vlmax = op_muli(vscale, baseVlmax_cIndex);

  VectorType outputMatTy = cast<VectorType>(dotOp.getC().getType());
  VectorType inputSubVecTy, outputSubVecTy;
  if (auto vscaleDefOp = vscale.getDefiningOp<arith::ConstantIndexOp>()) {
    inputSubVecTy = VectorType::get({baseVlmax_i64 * vscaleDefOp.value()},
                                    inputElemTy, {false});
    outputSubVecTy = VectorType::get({baseVlmax_i64 * vscaleDefOp.value()},
                                     outputElemTy, {false});
  } else {
    inputSubVecTy = VectorType::get({baseVlmax_i64}, inputElemTy, {true});
    outputSubVecTy = VectorType::get({baseVlmax_i64}, outputElemTy, {true});
  }

  Value resMat = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(outputMatTy));

  const int64_t MR = 4;
  const int64_t NR = 4;

  for (int64_t i_mr = 0; i_mr < mat_m; i_mr += MR) {
    for (int64_t i_nr = 0; i_nr < mat_n; i_nr += NR) {
      const int mr = std::min(MR, mat_m - i_mr);
      const int nr = std::min(NR, mat_n - i_nr);

      SmallVector<Value, MR * NR> sumVecs(
          mr * nr, arith::ConstantOp::create(
                       rewriter, loc, rewriter.getZeroAttr(outputSubVecTy)));

      auto genForBodyK = [&](Value iv, bool inBounds) {
        Value subVecOff = op_muli(iv, vlmax);
        // Get operands
        SmallVector<Value, MR> lhsVecs(mr);
        SmallVector<Value, NR> rhsVecs(nr);
        for (int64_t i_m = 0; i_m < mr; i_m++) {
          lhsVecs[i_m] =
              loadRow(rewriter, loc, lhsReadOp, index_cst(i_mr + i_m),
                      subVecOff, inputSubVecTy, inBounds);
        }
        for (int64_t i_n = 0; i_n < nr; i_n++) {
          rhsVecs[i_n] =
              loadCol(rewriter, loc, rhsReadOp, subVecOff,
                      index_cst(i_nr + i_n), inputSubVecTy, inBounds);
        }
        // Update sumVec
        SmallVector<Value, MR * NR> newSumVecs(mr * nr);
        for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
          for (int64_t i_nr = 0; i_nr < nr; i_nr++) {
            const int64_t id_region_iter = i_mr * nr + i_nr;
            // Prepare and call intrinsic
            Value sumVec = sumVecs[id_region_iter];
            Value lhsVec =
                maybeCast(loc, lhsVecs[i_mr], outputElemTy, rewriter);
            Value rhsVec =
                maybeCast(loc, rhsVecs[i_nr], outputElemTy, rewriter);
            Value newSumVec;
            if (outputElemTy.isInteger()) {
              auto mul = arith::MulIOp::create(rewriter, loc, lhsVec, rhsVec);
              newSumVec = arith::AddIOp::create(rewriter, loc, sumVec, mul);
            } else {
              newSumVec =
                  vector::FMAOp::create(rewriter, loc, lhsVec, rhsVec, sumVec);
            }
            newSumVecs[id_region_iter] = newSumVec;
          }
        }
        // Yield intrinsic result
        mlir::scf::YieldOp::create(rewriter, loc, newSumVecs);
      };

      // for k in [0, mat_k / vlmax * vlmax)
      Value numSubVec =
          arith::DivSIOp::create(rewriter, loc, index_cst(mat_k), vlmax);
      auto forOp = scf::ForOp::create(rewriter, loc, index_cst(0), numSubVec,
                                      index_cst(1), sumVecs);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        sumVecs.assign(forOp.getRegionIterArgs().begin(),
                       forOp.getRegionIterArgs().end());
        genForBodyK(forOp.getInductionVar(), true);
      }
      sumVecs = forOp.getResults();
      // for k in [mat_k / vlmax * vlmax, mat_k)
      Value kModVl =
          arith::RemSIOp::create(rewriter, loc, index_cst(mat_k), vlmax);
      auto ifOp = scf::IfOp::create(
          rewriter, loc,
          /*resultTypes=*/
          llvm::map_to_vector(sumVecs, [&](Value v) { return v.getType(); }),
          /*condition=*/
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, kModVl,
                                index_cst(0)),
          /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        genForBodyK(numSubVec, false);
      }
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        scf::YieldOp::create(rewriter, loc, sumVecs);
      }
      sumVecs = ifOp.getResults();

      // Reduction: Sum the accumulated results horizontally.
      for (int64_t i_m = 0; i_m < mr; i_m++) {
        for (int64_t i_n = 0; i_n < nr; i_n++) {
          const int64_t id_region_iter = i_m * nr + i_n;
          Value newRedSum;
          if (isAccZeroInit) {
            newRedSum = vector::ReductionOp::create(rewriter, loc,
                                                    vector::CombiningKind::ADD,
                                                    sumVecs[id_region_iter]);
          } else {
            SmallVector<Value> indices = accBuf.indices;
            indices[0] = arith::AddIOp::create(rewriter, loc, indices[0],
                                               index_cst(i_mr + i_m));
            indices[1] = arith::AddIOp::create(rewriter, loc, indices[1],
                                               index_cst(i_nr + i_n));
            Value redsum =
                memref::LoadOp::create(rewriter, loc, accBuf.memRef, indices);
            newRedSum = vector::ReductionOp::create(
                rewriter, loc, vector::CombiningKind::ADD,
                sumVecs[id_region_iter], redsum);
          }
          resMat = vector::InsertOp::create(
              rewriter, loc, newRedSum, resMat,
              SmallVector<int64_t>({i_mr + i_m, i_nr + i_n}));
        }
      }
    }
  }

  rewriter.replaceOp(dotOp, resMat);
  return success();
}

vector::TransferReadOp storeToTmpBufferAndRead(PatternRewriter &rewriter,
                                               Location loc,
                                               Operation *allocaPoint,
                                               cpu::DotOp dotOp, Value mat) {
  auto matTy = cast<VectorType>(mat.getType());
  auto buffer = allocateTmpBufferStack(loc, matTy, allocaPoint, rewriter);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(dotOp);
  auto writeOp = vector::TransferWriteOp::create(rewriter, loc, mat,
                                                 buffer.memRef, buffer.indices);
  auto readOp = vector::TransferReadOp::create(
      rewriter, loc, matTy, buffer.memRef, buffer.indices, std::nullopt);

  return readOp;
}

LogicalResult convertRvvCandidate(RvvDotOpCandidate &candidate,
                                  PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  if (!candidate.lhsReadOp) {
    candidate.lhsReadOp =
        storeToTmpBufferAndRead(rewriter, loc, allocaPoint, op, op.getA());
  }

  if (!candidate.rhsReadOp) {
    candidate.rhsReadOp =
        storeToTmpBufferAndRead(rewriter, loc, allocaPoint, op, op.getB());
  }

  Value acc = op.getC();
  bool isAccZeroInit = isZeroConst(acc);
  MemBuffer accBuf;
  if (isAccZeroInit) {
    accBuf = allocateTmpBufferStack(loc, cast<VectorType>(acc.getType()),
                                    allocaPoint, rewriter);
  } else {
    accBuf = storeToTmpBuffer(loc, acc, allocaPoint, rewriter);
  }

  // lower dot op in the specified style
  switch (candidate.dotStyle) {
  case OUTER: {
    return convertToOuterProductGemm(candidate, rewriter, accBuf,
                                     isAccZeroInit);
  }
  case INNER: {
    return convertToInnerProductGemm(candidate, rewriter, accBuf,
                                     isAccZeroInit);
  }
  default: {
    llvm_unreachable("Unknown dot style.");
  }
  }
}

struct ConvertDotToRVV
    : public triton::cpu::impl::ConvertDotToRVVBase<ConvertDotToRVV> {
  ConvertDotToRVV() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<RvvDotOpCandidate, 1> candidates;
    mod->walk([this, &candidates](cpu::DotOp op) {
      RvvDotOpCandidate candidate;
      if (isRvvCandidate(op, candidate)) {
        LLVM_DEBUG({
          LDBG("Found RVV candidate");
          LDBG("  Op: " << candidate.op);
          LDBG("  DotStyle: " << (candidate.dotStyle == INNER ? "INNER"
                                                              : "OUTER"));
          LDBG("  InputElemTy: " << candidate.inputElemTy);
          LDBG("  OutputElemTy: " << candidate.outputElemTy);
          if (candidate.lhsReadOp) {
            LDBG("  LhsReadOp: " << candidate.lhsReadOp);
          }
          if (candidate.rhsReadOp) {
            LDBG("  RhsReadOp: " << candidate.rhsReadOp);
          }
        });
        candidates.push_back(candidate);
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate.op);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(convertRvvCandidate(candidate, rewriter))) {
        LDBG("Conversion succeeded!");
      } else {
        LDBG("Conversion failed!");
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToRVV() {
  return std::make_unique<ConvertDotToRVV>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir