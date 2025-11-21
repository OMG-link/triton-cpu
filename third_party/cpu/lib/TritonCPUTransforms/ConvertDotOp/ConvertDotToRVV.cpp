#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

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
  // True if outputElemTy.bitwidth = inputElemTy.bitwidth * 2
  bool isWidening;

  // Matrix sizes. LHS = <m x k>; RHS = <k x n>; output = <m x n>
  int64_t m, k, n;

  // The way we do GEMM. Support: outer-product(OUTER), inner-product(INNER)
  DotStyle dotStyle;

  // Memory buffer holding LHS. Can be empty if LHS is not a result of a
  // simple load.
  MemBuffer lhsBuf;
  // Memory buffer holding RHS. Can be empty if RHS is not a result of a
  // simple load.
  MemBuffer rhsBuf;
};

// Check if input types are same, and if output elemets types are same with
// input or double-width relative to input. If success, inputElemTy and
// isWidening in candidate are filled.
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

  // isWidening
  if (inputElemTy.getIntOrFloatBitWidth() ==
      outputElemTy.getIntOrFloatBitWidth()) {
    candidate.isWidening = false;
  } else if (inputElemTy.getIntOrFloatBitWidth() * 2 ==
             outputElemTy.getIntOrFloatBitWidth()) {
    candidate.isWidening = true;
  } else {
    LDBG("checkElemTypes failed: bitwidth of output is neither same nor double "
         "of input. (inputElemTy)"
         << inputElemTy << " (outputElemTy)" << outputElemTy);
    return false;
  }

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

// Returns the first dimension with stride 1, which is usually considered as
// 'lowest' dimention.
int64_t findLowestDim(MemRefType memRefType) {
  llvm::SmallVector<int64_t, 8> strides;
  int64_t offset = 0;
  if (succeeded(memRefType.getStridesAndOffset(strides, offset))) {
    // Find the dimension with stride 1
    int64_t lowestDim = 0;
    for (size_t i = 0; i < strides.size(); ++i) {
      int64_t s = strides[i];
      if (s == 1) {
        lowestDim = static_cast<int64_t>(i);
        break;
      }
    }
    return lowestDim;
  } else {
    // Failed to get strides? Don't know what happened.
    // Return the last dimension as default.
    return memRefType.getRank() - 1;
  }
}

int64_t findLowestDim(const MemBuffer &buf) {
  if (buf.empty()) {
    // storeToTempBuffer will make the last dimension continuous.
    return static_cast<int64_t>(buf.indices.size()) - 1;
  }

  auto memRefType = dyn_cast<MemRefType>(buf.memRef.getType());
  if (!memRefType) {
    // Don't know how to find lowest dimension if memref is not memref.
    return static_cast<int64_t>(buf.indices.size()) - 1;
  }

  int64_t lowestDim = findLowestDim(memRefType);
  int64_t totalDim = memRefType.getRank();

  // Process buf.transposed
  if (lowestDim == totalDim - 1 && buf.transposed) {
    lowestDim = totalDim - 2;
  } else if (lowestDim == totalDim - 2 && buf.transposed) {
    lowestDim = totalDim - 1;
  }

  return lowestDim;
}

// Determine dot style by input data layout.
void determineDotStyle(Value a, Value b, RvvDotOpCandidate &candidate) {
  candidate.lhsBuf = findInputBuffer(a, true);
  candidate.rhsBuf = findInputBuffer(b, true);
  int64_t lhsLowestDim =
      findLowestDim(candidate.lhsBuf) - candidate.lhsBuf.indices.size();
  int64_t rhsLowestDim =
      findLowestDim(candidate.rhsBuf) - candidate.rhsBuf.indices.size();
  if (rhsLowestDim == -1) {
    // When the last dimension of right operand(N) is continuous, we use
    // outer-product GEMM.
    LDBG("Last dimension of right operand is continuous. "
         "Recommend outer-product GEMM.");
    candidate.dotStyle = OUTER;
  } else if (rhsLowestDim == -2 && lhsLowestDim == -1) {
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

// Get RVV VSCALE.
//
// VSCALE is a implement-defined value. Program can only know that value at
// runtime. In triton, we use JIT, so we may be possible to get VSCALE when
// compiling. We support three ways to specific VSCALE using environment
// variable RVV_VLEN:
//
// 1) When RVV_VLEN is an integer, compiler uses RVV_VLEN/64 as VSCALE. (If it
// is a valid VLEN: RVV_VLEN is power of 2 && RVV_VLEN >= 64)
// 2) When RVV_VLEN is "dynamic", compiler uses a runtime call @vector.vscale to
// get VSCALE.
// 3) When RVV_VLEN is "local", compiler executes assembly `csrr %0, vlenb` to
// get VSCALE.
//
// If RVV_VLEN is undefined or invalid(including RVV_VLEN is "local" but the
// compiler is running on a machine that doesn't support RISCV-V-Extension),
// compiler will fallback and tries the following ways one by one:
// - Executes assembly `csrr %0, vlenb` to get VSCALE if possible.
// - Return a runtime call @vector.vscale.

static inline int64_t tryReadVlenb() {
#ifdef __riscv_vector
  int64_t vlenb;
  asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  return vlenb;
#else
  return -1;
#endif
}

int64_t getVlen() {
  std::string RVV_VLEN = mlir::triton::tools::getStrEnv("RVV_VLEN");
  // if RVV_VLEN is defined
  if (!RVV_VLEN.empty()) {
    if (RVV_VLEN == "dynamic") {
      return -1;
    } else if (RVV_VLEN == "local") {
      int vlenb = tryReadVlenb();
      if (vlenb > 0) {
        return vlenb * 8;
      } else {
        return -1;
      }
    } else {
      char *end;
      long vlen = strtol(RVV_VLEN.c_str(), &end, 10);
      if (*end == '\0') {
        if (vlen >= 64 && (vlen & (vlen - 1)) == 0) {
          return vlen;
        }
      }
    }
  }
  // fallback
  int vlenb = tryReadVlenb();
  if (vlenb > 0) {
    return vlenb * 8;
  }
  return -1;
}

// Returns: VSCALE of type 'index'
Value getVscale(Location loc, PatternRewriter &rewriter) {
  int vlen = getVlen();
  if (vlen > 0) {
    return rewriter.create<arith::ConstantIndexOp>(loc, vlen / 64);
  } else {
    return rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
  }
}

SmallVector<Value> shiftIndices(Location loc, ArrayRef<Value> indices,
                                bool transposed, Value m, Value n,
                                PatternRewriter &rewriter) {
  SmallVector<Value> res(indices.begin(), indices.end() - 2);
  if (transposed)
    std::swap(m, n);
  res.push_back(op_addi(*(indices.end() - 2), m));
  res.push_back(op_addi(*(indices.end() - 1), n));
  return res;
}

SmallVector<Value> shiftIndices(Location loc, const MemBuffer &buf, Value m,
                                Value n, PatternRewriter &rewriter) {
  return shiftIndices(loc, buf.indices, buf.transposed, m, n, rewriter);
}

Value loadScalar(Location loc, PatternRewriter &rewriter, const MemBuffer &buf,
                 Value m, Value n) {
  SmallVector<Value> indices = shiftIndices(loc, buf, m, n, rewriter);
  return rewriter.create<memref::LoadOp>(loc, buf.memRef, indices);
}

// Load vector at memRef[indices].
// Result vector has type resTy, and will be read along the resDim-th dimesion.
Value loadVec(Location loc, PatternRewriter &rewriter, VectorType resTy,
              int64_t resDim, Value resLen, const Value &memRef,
              ValueRange indices) {
  assert(resTy.getRank() == 1);
  AffineExpr resDimAffineExpr = rewriter.getAffineDimExpr(resDim);
  AffineMap affineMap = AffineMap::get(indices.size(), 0, resDimAffineExpr);
  Value padding = rewriter.create<LLVM::UndefOp>(loc, resTy.getElementType());
  VectorType maskType = resTy.cloneWith(std::nullopt, rewriter.getI1Type());
  Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, resLen);
  ArrayAttr inBounds = rewriter.getBoolArrayAttr({true});
  return rewriter.create<vector::TransferReadOp>(
      loc, resTy, memRef, indices, affineMap, padding, mask, inBounds);
}

Value loadRow(Location loc, PatternRewriter &rewriter, VectorType resTy,
              Value resLen, const MemBuffer &buf, const Value &off2,
              const Value &off1) {
  assert(buf.indices.size() >= 2);
  int64_t resDim;
  SmallVector<Value> indices = buf.indices;
  if (buf.transposed) {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off2);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off1);
    resDim = static_cast<int64_t>(buf.indices.size()) - 2;
  } else {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off1);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off2);
    resDim = static_cast<int64_t>(buf.indices.size()) - 1;
  }
  return loadVec(loc, rewriter, resTy, resDim, resLen, buf.memRef, indices);
}

Value loadCol(Location loc, PatternRewriter &rewriter, VectorType resTy,
              Value resLen, const MemBuffer &buf, const Value &off2,
              const Value &off1) {
  assert(buf.indices.size() >= 2);
  int64_t resDim;
  SmallVector<Value> indices = buf.indices;
  if (buf.transposed) {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off2);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off1);
    resDim = static_cast<int64_t>(buf.indices.size()) - 1;
  } else {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off1);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off2);
    resDim = static_cast<int64_t>(buf.indices.size()) - 2;
  }
  return loadVec(loc, rewriter, resTy, resDim, resLen, buf.memRef, indices);
}

SmallVector<Value> loadRows(Location loc, VectorType rowTy, int64_t rowNum,
                            Value colNum, const MemBuffer &buf,
                            const Value &subVecOff, PatternRewriter &rewriter) {
  SmallVector<Value> vecs;
  vecs.reserve(rowNum);
  for (int64_t m = 0; m < rowNum; ++m) {
    Value cIndex_m = index_cst(m);
    vecs.push_back(
        loadRow(loc, rewriter, rowTy, colNum, buf, cIndex_m, subVecOff));
  }
  return vecs;
}

void storeVec(Location loc, PatternRewriter &rewriter, Value vec,
              int64_t resDim, Value resLen, const Value &memRef,
              ValueRange indices) {
  VectorType vecTy = cast<VectorType>(vec.getType());
  assert(vecTy.getRank() == 1);
  AffineExpr resDimAffineExpr = rewriter.getAffineDimExpr(resDim);
  AffineMap affineMap = AffineMap::get(indices.size(), 0, resDimAffineExpr);
  AffineMapAttr affineMapAttr = AffineMapAttr::get(affineMap);
  ArrayAttr inBounds = rewriter.getBoolArrayAttr({true});
  VectorType maskType = vecTy.cloneWith(std::nullopt, rewriter.getI1Type());
  Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, resLen);
  rewriter.create<vector::TransferWriteOp>(loc, vec, memRef, indices,
                                           affineMapAttr, mask, inBounds);
}

void storeRow(Location loc, PatternRewriter &rewriter, Value vec, Value resLen,
              const MemBuffer &buf, const Value &off2, const Value &off1) {
  assert(buf.indices.size() >= 2);
  int64_t resDim;
  SmallVector<Value> indices = buf.indices;
  if (buf.transposed) {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off2);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off1);
    resDim = static_cast<int64_t>(buf.indices.size()) - 2;
  } else {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off1);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off2);
    resDim = static_cast<int64_t>(buf.indices.size()) - 1;
  }
  storeVec(loc, rewriter, vec, resDim, resLen, buf.memRef, indices);
}

void storeRows(Location loc, const MemBuffer &buf, ArrayRef<Value> vecs,
               Value colNum, const Value &subVecOff,
               PatternRewriter &rewriter) {
  for (size_t m = 0; m < vecs.size(); ++m) {
    Value cIndex_m = index_cst(static_cast<int64_t>(m));
    storeRow(loc, rewriter, vecs[m], colNum, buf, cIndex_m, subVecOff);
  }
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
  if (auto vlen = getVlen(); vlen > 0) {
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
      return rewriter.create<arith::ExtSIOp>(loc, dstTy, val);
    return rewriter.create<arith::TruncIOp>(loc, dstTy, val);
  } else {
    if (srcElemTy.getIntOrFloatBitWidth() < dstElemTy.getIntOrFloatBitWidth())
      return rewriter.create<arith::ExtFOp>(loc, dstTy, val);
    return rewriter.create<arith::TruncFOp>(loc, dstTy, val);
  }
}

LogicalResult convertToOuterProductGemm(RvvDotOpCandidate &candidate,
                                        PatternRewriter &rewriter,
                                        MemBuffer accBuf, bool isAccZeroInit) {
  cpu::DotOp dotOp = candidate.op;
  MemBuffer lhsBuf = candidate.lhsBuf;
  MemBuffer rhsBuf = candidate.rhsBuf;
  Type inputElemTy = candidate.inputElemTy;
  Type outputElemTy = candidate.outputElemTy;
  int64_t mat_m = candidate.m;
  int64_t mat_n = candidate.n;
  int64_t mat_k = candidate.k;
  bool isWidening = candidate.isWidening;

  Location loc = dotOp.getLoc();

  int64_t outputElemBitWidth = outputElemTy.getIntOrFloatBitWidth();
  // If mat_k==1, the result is available immediately. No accumulation is
  // needed.
  int64_t vmul = getVmul(/*numAcc=*/mat_k > 1 ? mat_m : 1,
                         /*accBits=*/mat_n * outputElemBitWidth);
  const int64_t baseVlen = 64;
  const int64_t baseVlmax = vmul * baseVlen / outputElemBitWidth;

  Value baseVlmax_cIndex = index_cst(baseVlmax);

  Value vscale = getVscale(loc, rewriter);
  Value vlmax = op_muli(vscale, baseVlmax_cIndex);

  VectorType outputMatTy = cast<VectorType>(dotOp.getC().getType());
  VectorType inputSubVecTy, outputSubVecTy;
  if (auto vscaleDefOp = vscale.getDefiningOp<arith::ConstantIndexOp>()) {
    inputSubVecTy = VectorType::get({baseVlmax * vscaleDefOp.value()},
                                    inputElemTy, {false});
    outputSubVecTy = VectorType::get({baseVlmax * vscaleDefOp.value()},
                                     outputElemTy, {false});
  } else {
    inputSubVecTy = VectorType::get({baseVlmax}, inputElemTy, {true});
    outputSubVecTy = VectorType::get({baseVlmax}, outputElemTy, {true});
  }

  // We will do GEMM by multiplying an <M x 1> vector with an <1 x N> vector.
  // To do so, we multiply <1 x 1> scalar with <1 x N> vector.
  // As N is given by the user, it can be too large for single vector register.
  // We need to divide <1 x N> vector into <1 x VL> one, where VL is the number
  // of elements single vector register can hold.

  // Code generator that computes <M x K> x <K x VL>.
  auto genForBodyN = [&](Value iv, Value vl) {
    Value subVecOff = op_muli(iv, vlmax);

    SmallVector<Value> accVecs;
    if (isAccZeroInit) {
      accVecs.resize(mat_m);
    } else {
      accVecs =
          loadRows(loc, outputSubVecTy, mat_m, vl, accBuf, subVecOff, rewriter);
    }

    auto doMul = [&](Value lhsScalar, Value rhsVec) -> Value {
      if (lhsScalar.getType().isIntOrFloat()) {
        lhsScalar = maybeCast(loc, lhsScalar, outputElemTy, rewriter);
        rhsVec = maybeCast(loc, rhsVec, outputElemTy, rewriter);
        auto splat =
            rewriter.create<vector::SplatOp>(loc, rhsVec.getType(), lhsScalar);
        if (lhsScalar.getType().isInteger()) {
          return rewriter.create<arith::MulIOp>(loc, splat, rhsVec);
        } else {
          return rewriter.create<arith::MulFOp>(loc, splat, rhsVec,
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
        auto splat =
            rewriter.create<vector::SplatOp>(loc, rhsVec.getType(), lhsScalar);
        if (lhsScalar.getType().isInteger()) {
          auto mul = rewriter.create<arith::MulIOp>(loc, splat, rhsVec);
          return rewriter.create<arith::AddIOp>(loc, accVec, mul);
        } else {
          return rewriter.create<vector::FMAOp>(loc, splat, rhsVec, accVec);
        }
      } else {
        // report type of lhsScalar is unexpected
        llvm_unreachable("Unexpected type of lhsScalar in doMacc.");
      }
    };

    // k = 0
    {
      Value cIndex_k = index_cst(0);
      Value rhsVec = loadRow(loc, rewriter, inputSubVecTy, vl, rhsBuf, cIndex_k,
                             subVecOff);
      for (int64_t m = 0; m < mat_m; ++m) {
        Value lhsScalar =
            loadScalar(loc, rewriter, lhsBuf, index_cst(m), cIndex_k);
        if (isAccZeroInit) {
          accVecs[m] = doMul(lhsScalar, rhsVec);
        } else {
          accVecs[m] = doMacc(accVecs[m], lhsScalar, rhsVec);
        }
      }
    }

    // for k in [1, mat_k)
    auto forOpK = rewriter.create<scf::ForOp>(
        loc, index_cst(1), index_cst(mat_k), index_cst(1), accVecs);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOpK.getBody());

      Value cIndex_k = forOpK.getInductionVar();
      Value rhsVec = loadRow(loc, rewriter, inputSubVecTy, vl, rhsBuf, cIndex_k,
                             subVecOff);

      SmallVector<Value> accVecs(mat_m);
      for (int64_t m = 0; m < mat_m; ++m) {
        Value lhsScalar =
            loadScalar(loc, rewriter, lhsBuf, index_cst(m), cIndex_k);
        accVecs[m] = doMacc(forOpK.getRegionIterArg(m), lhsScalar, rhsVec);
      }
      rewriter.create<scf::YieldOp>(loc, accVecs);
    } // end of for-op-k
    accVecs = forOpK.getResults();
    storeRows(loc, accBuf, accVecs, vl, subVecOff, rewriter);
  };

  // Divide <K x N> vector into <K x VL> ones.
  // We do this first to achieve the best performance.
  Value numSubVec =
      rewriter.create<arith::DivSIOp>(loc, index_cst(mat_n), vlmax);
  auto forOpN =
      rewriter.create<scf::ForOp>(loc, index_cst(0), numSubVec, index_cst(1));
  // Process each <M x K> x <K x VL> sub-matrix-product.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOpN.getBody());
    genForBodyN(forOpN.getInductionVar(), vlmax);
  }
  // Process the remaining <K x (N % VL)> vector if N is not multiple of VL.
  Value nModVl = rewriter.create<arith::RemSIOp>(loc, index_cst(mat_n), vlmax);
  auto ifOp = rewriter.create<scf::IfOp>(
      loc,
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nModVl,
                                     index_cst(0)),
      /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.getBody());
    genForBodyN(numSubVec, nModVl);
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
  MemBuffer lhsBuf = candidate.lhsBuf;
  MemBuffer rhsBuf = candidate.rhsBuf;
  Type inputElemTy = candidate.inputElemTy;
  Type outputElemTy = candidate.outputElemTy;
  int64_t mat_m = candidate.m;
  int64_t mat_n = candidate.n;
  int64_t mat_k = candidate.k;
  bool isWidening = candidate.isWidening;

  Location loc = dotOp.getLoc();

  int64_t inputElemBitWidth = inputElemTy.getIntOrFloatBitWidth();
  const int64_t baseVlen_i64 = 64;
  const int64_t baseVlmax_i64 =
      baseVlen_i64 / inputElemBitWidth / (isWidening ? 2 : 1);
  assert(!(inputElemBitWidth == 64 && isWidening) &&
         "Cannot widen 64-bit element.");
  assert(baseVlmax_i64 > 0);

  Value baseVlmax_cIndex = index_cst(baseVlmax_i64);

  Value vscale = getVscale(loc, rewriter);
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

  Value resMat = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(outputMatTy));

  const int64_t MR = 4;
  const int64_t NR = 4;

  for (int64_t m = 0; m < mat_m; m += MR) {
    for (int64_t n = 0; n < mat_n; n += NR) {
      const int mr = std::min(MR, mat_m - m);
      const int nr = std::min(NR, mat_n - n);

      SmallVector<Value, MR * NR> sumVecs(
          mr * nr, rewriter.create<arith::ConstantOp>(
                       loc, rewriter.getZeroAttr(outputSubVecTy)));

      auto genForBodyK = [&](Value iv, Value vl) {
        Value subVecOff = op_muli(iv, vlmax);
        // Get operands
        SmallVector<Value, MR> lhsVecs(mr);
        SmallVector<Value, NR> rhsVecs(nr);
        for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
          lhsVecs[i_mr] = loadRow(loc, rewriter, inputSubVecTy, vl, lhsBuf,
                                  index_cst(m + i_mr), subVecOff);
        }
        for (int64_t i_nr = 0; i_nr < nr; i_nr++) {
          rhsVecs[i_nr] = loadCol(loc, rewriter, inputSubVecTy, vl, rhsBuf,
                                  subVecOff, index_cst(n + i_nr));
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
              auto mul = rewriter.create<arith::MulIOp>(loc, lhsVec, rhsVec);
              newSumVec = rewriter.create<arith::AddIOp>(loc, sumVec, mul);
            } else {
              newSumVec =
                  rewriter.create<vector::FMAOp>(loc, lhsVec, rhsVec, sumVec);
            }
            newSumVecs[id_region_iter] = newSumVec;
          }
        }
        // Yield intrinsic result
        rewriter.create<mlir::scf::YieldOp>(loc, newSumVecs);
      };

      // for k in [0, mat_k / vlmax * vlmax)
      Value numSubVec =
          rewriter.create<arith::DivSIOp>(loc, index_cst(mat_k), vlmax);
      auto forOp = rewriter.create<scf::ForOp>(loc, index_cst(0), numSubVec,
                                               index_cst(1), sumVecs);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        sumVecs.assign(forOp.getRegionIterArgs().begin(),
                       forOp.getRegionIterArgs().end());
        genForBodyK(forOp.getInductionVar(), vlmax);
      }
      sumVecs = forOp.getResults();
      // for k in [mat_k / vlmax * vlmax, mat_k)
      Value kModVl =
          rewriter.create<arith::RemSIOp>(loc, index_cst(mat_k), vlmax);
      auto ifOp = rewriter.create<scf::IfOp>(
          loc,
          /*resultTypes=*/
          llvm::map_to_vector(sumVecs, [&](Value v) { return v.getType(); }),
          /*condition=*/
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, kModVl,
                                         index_cst(0)),
          /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        genForBodyK(numSubVec, kModVl);
      }
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        rewriter.create<scf::YieldOp>(loc, sumVecs);
      }
      sumVecs = ifOp.getResults();

      // Reduction: Sum the accumulated results horizontally.
      for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
        for (int64_t i_nr = 0; i_nr < nr; i_nr++) {
          const int64_t id_region_iter = i_mr * nr + i_nr;
          Value newRedSum;
          if (isAccZeroInit) {
            newRedSum = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, sumVecs[id_region_iter]);
          } else {
            Value redsum = loadScalar(loc, rewriter, accBuf,
                                      index_cst(m + i_mr), index_cst(n + i_nr));
            newRedSum = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, sumVecs[id_region_iter],
                redsum);
          }
          resMat = rewriter.create<vector::InsertOp>(
              loc, newRedSum, resMat,
              SmallVector<int64_t>({m + i_mr, n + i_nr}));
        }
      }
    }
  }

  rewriter.replaceOp(dotOp, resMat);
  return success();
}

LogicalResult convertRvvCandidate(RvvDotOpCandidate &candidate,
                                  PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  if (candidate.lhsBuf.empty()) {
    Value lhs = op.getA();
    candidate.lhsBuf = storeToTmpBuffer(loc, lhs, allocaPoint, rewriter);
  }

  if (candidate.rhsBuf.empty()) {
    Value rhs = op.getB();
    candidate.rhsBuf = storeToTmpBuffer(loc, rhs, allocaPoint, rewriter);
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
          LDBG("  IsWidening: " << candidate.isWidening);
          if (!candidate.lhsBuf.empty()) {
            LDBG("  LhsBuf: " << candidate.lhsBuf.memRef);
            LDBG("  Transposed: " << candidate.lhsBuf.transposed);
          }
          if (!candidate.rhsBuf.empty()) {
            LDBG("  RhsBuf: " << candidate.rhsBuf.memRef);
            LDBG("  Transposed: " << candidate.rhsBuf.transposed);
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