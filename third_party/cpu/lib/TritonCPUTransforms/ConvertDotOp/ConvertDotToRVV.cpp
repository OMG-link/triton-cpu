#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

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

  if (resTy.getDimSize(1) < 8) {
    LDBG("checkInputShapes failed: resTy of length "
         << resTy.getDimSize(1) << " is too short for vectorize");
    return false;
  }

  // Fillin matrix size
  candidate.m = resTy.getDimSize(0);
  candidate.n = resTy.getDimSize(1);
  candidate.k = lhsTy.getDimSize(1);

  return true;
}

// Determine dot style by input data layout.
void determineDotStyle(Value a, Value b, RvvDotOpCandidate &candidate) {
  candidate.lhsBuf = findInputBuffer(a, true);
  candidate.rhsBuf = findInputBuffer(b, true);
  // FIXME:
  // MemBuffer.transposed方法只能说明findInputBuffer在寻找内存地址的过程中是否
  // 经历了转置，不代表矩阵在内存中被转置存储。
  if (candidate.lhsBuf.transposed) {
    if (candidate.rhsBuf.transposed) {
      LDBG("Both lhs and rhs are transposed. Nothing to recommend.");
      candidate.dotStyle = INNER;
    } else {
      LDBG("Only lhs is transposed. Recommend outer-product GEMM.");
      candidate.dotStyle = OUTER;
    }
  } else {
    if (candidate.rhsBuf.transposed) {
      LDBG("Only rhs is transposed. Recommend inner-product GEMM.");
      candidate.dotStyle = INNER;
    } else {
      LDBG("No operand transposed. Nothing to recommend.");
      candidate.dotStyle = INNER;
    }
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

// Returns: VSCALE of type 'index'
Value getVscale(Location loc, PatternRewriter &rewriter) {
  std::string RVV_VLEN = mlir::triton::tools::getStrEnv("RVV_VLEN");
  // if RVV_VLEN is defined
  if (!RVV_VLEN.empty()) {
    if (RVV_VLEN == "dynamic") {
      return rewriter.create<vector::VectorScaleOp>(loc,
                                                    rewriter.getIndexType());
    } else if (RVV_VLEN == "local") {
      int vlenb = tryReadVlenb();
      if (vlenb > 0) {
        return index_cst(vlenb / 8);
      } else {
        emitWarning(loc,
                    "Environment variable RVV_VLEN is set to 'local', but "
                    "triton compiler is not compiled with RISCV-V-Extension.");
      }
    } else {
      char *end;
      long vlen = strtol(RVV_VLEN.c_str(), &end, 10);
      if (*end == '\0') {
        if (vlen >= 64 && (vlen & (vlen - 1)) == 0) {
          return index_cst(vlen / 64);
        }
      }
      emitWarning(loc) << "Invalid RVV_VLEN option: " << RVV_VLEN;
    }
  }
  // fallback
  int vlenb = tryReadVlenb();
  if (vlenb > 0) {
    return index_cst(vlenb / 8);
  }
  return rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
}

SmallVector<Value> shiftIndices(Location loc, ArrayRef<Value> indices,
                                bool transposed, int64_t m, int64_t n,
                                PatternRewriter &rewriter) {
  SmallVector<Value> res(indices.begin(), indices.end() - 2);
  if (transposed)
    std::swap(m, n);
  res.push_back(shiftIndex(loc, *(indices.end() - 2), m, rewriter));
  res.push_back(shiftIndex(loc, *(indices.end() - 1), n, rewriter));
  return res;
}

SmallVector<Value> shiftIndices(Location loc, const MemBuffer &buf, int64_t m,
                                int64_t n, PatternRewriter &rewriter) {
  return shiftIndices(loc, buf.indices, buf.transposed, m, n, rewriter);
}

Value loadScalar(Location loc, PatternRewriter &rewriter, const MemBuffer &buf,
                 int64_t m, int64_t n) {
  SmallVector<Value> indices = shiftIndices(loc, buf, m, n, rewriter);
  return rewriter.create<memref::LoadOp>(loc, buf.memRef, indices);
}

// Load vector at memRef[indices].
// Result vector has type resTy, and will be read along the resDim-th dimesion.
Value loadVec(Location loc, PatternRewriter &rewriter, VectorType resTy,
              int64_t resDim, const Value &memRef, ValueRange indices) {
  assert(resTy.getRank() == 1);
  AffineExpr resDimAffineExpr = rewriter.getAffineDimExpr(resDim);
  AffineMap affineMap = AffineMap::get(indices.size(), 0, resDimAffineExpr);
  return rewriter.create<vector::TransferReadOp>(loc, resTy, memRef, indices,
                                                 affineMap);
}

Value loadRow(Location loc, PatternRewriter &rewriter, VectorType resTy,
              const MemBuffer &buf, const Value &off2, const Value &off1) {
  assert(buf.indices.size() >= 2);
  int64_t vecDim;
  SmallVector<Value> indices = buf.indices;
  if (buf.transposed) {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off2);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off1);
    vecDim = static_cast<int64_t>(buf.indices.size()) - 2;
  } else {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off1);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off2);
    vecDim = static_cast<int64_t>(buf.indices.size()) - 1;
  }
  return loadVec(loc, rewriter, resTy, vecDim, buf.memRef, indices);
}

Value loadCol(Location loc, PatternRewriter &rewriter, VectorType resTy,
              const MemBuffer &buf, const Value &off2, const Value &off1) {
  assert(buf.indices.size() >= 2);
  int64_t vecDim;
  SmallVector<Value> indices = buf.indices;
  if (buf.transposed) {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off2);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off1);
    vecDim = static_cast<int64_t>(buf.indices.size()) - 1;
  } else {
    indices[indices.size() - 1] = op_addi(indices[indices.size() - 1], off1);
    indices[indices.size() - 2] = op_addi(indices[indices.size() - 2], off2);
    vecDim = static_cast<int64_t>(buf.indices.size()) - 2;
  }
  return loadVec(loc, rewriter, resTy, vecDim, buf.memRef, indices);
}

SmallVector<Value> loadRows(Location loc, VectorType rowTy, int64_t rowNum,
                            const MemBuffer &buf, const Value &subVecOff,
                            PatternRewriter &rewriter) {
  SmallVector<Value> vecs;
  vecs.reserve(rowNum);
  for (int64_t m = 0; m < rowNum; ++m) {
    Value cIndex_m = index_cst(m);
    vecs.push_back(loadRow(loc, rewriter, rowTy, buf, cIndex_m, subVecOff));
  }
  return vecs;
}

void storeRow(Location loc, const MemBuffer &buf, int64_t rowIdx, Value vec,
              const Value &subVecOff, PatternRewriter &rewriter) {
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, buf.indices[indices.size() - 2], rowIdx, rewriter);
  indices[indices.size() - 1] = subVecOff;
  rewriter.create<vector::StoreOp>(loc, vec, buf.memRef, indices);
}

void storeRows(Location loc, const MemBuffer &buf,
               const SmallVector<Value> &vecs, const Value &subVecOff,
               PatternRewriter &rewriter) {
  for (int64_t m = 0; m < vecs.size(); ++m)
    storeRow(loc, buf, m, vecs[m], subVecOff, rewriter);
}

StringAttr getMulIntrinsicName(PatternRewriter &rewriter, StringRef opName,
                               bool isFloat, bool isWidening) {
  std::string name = llvm::formatv("llvm.riscv.v{0}{1}{2}",
                                   isFloat ? "f" : "",    // 0
                                   isWidening ? "w" : "", // 1
                                   opName);               // 2
  return rewriter.getStringAttr(name);
}

StringAttr getReduceIntrinsicName(PatternRewriter &rewriter, StringRef opName,
                                  bool isFloat, StringRef fOrder = "") {
  std::string name = llvm::formatv("llvm.riscv.v{0}red{1}{2}",
                                   isFloat ? "f" : "",    // 0
                                   isFloat ? fOrder : "", // 1
                                   opName);               // 2
  return rewriter.getStringAttr(name);
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

  int64_t inputElemBitWidth = inputElemTy.getIntOrFloatBitWidth();
  const int64_t baseVlen = 64;
  const int64_t baseInputVl = baseVlen / inputElemBitWidth;

  VectorType outputMatTy = cast<VectorType>(dotOp.getC().getType());
  VectorType inputSubVecTy =
      VectorType::get({baseInputVl}, inputElemTy, {true});
  VectorType outputSubVecTy =
      VectorType::get({baseInputVl}, outputElemTy, {true});

  Value c_tumu = int_cst(rewriter.getI64Type(), 3);
  Value c_frm_dyn = int_cst(rewriter.getI64Type(), 7);
  Value cIndex_0 = index_cst(0);
  Value cIndex_1 = index_cst(1);
  Value cIndex_baseVl = index_cst(baseInputVl);
  Value cIndex_n = index_cst(mat_n);

  // We will do GEMM by multiplying an <M x 1> vector with an <1 x N> vector.
  // To do so, we multiply <1 x 1> scalar with <1 x N> vector.
  // As N is given by the user, it can be too large for single vector register.
  // We need to divide <1 x N> vector into <1 x VL> one, where VL is the number
  // of elements single vector register can hold.

  // Divide <1 x N> vector into <1 x VL> one.
  // We do this first to achieve the best performance.
  Value vscale = getVscale(loc, rewriter);
  Value vl = op_muli(vscale, cIndex_baseVl);
  Value numSubVec = rewriter.create<arith::CeilDivSIOp>(loc, cIndex_n, vl);
  auto forOp = rewriter.create<scf::ForOp>(loc, cIndex_0, numSubVec, cIndex_1);

  // For-op body: this for-op divide <* x N> vector into <* x VL> one
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    // curVl = min(vl, n - iv * vl)
    Value iv = forOp.getInductionVar();
    Value subVecOff = op_muli(iv, vl);
    Value remaining = op_subi(cIndex_n, subVecOff);
    Value curVl = op_index_cast(rewriter.getI64Type(), op_minui(vl, remaining));

    SmallVector<Value> accVecs;
    if (isAccZeroInit) {
      accVecs.reserve(mat_m);
    } else {
      accVecs =
          loadRows(loc, outputSubVecTy, mat_m, accBuf, subVecOff, rewriter);
    }

    Value nextRhsVec =
        loadRow(loc, rewriter, inputSubVecTy, rhsBuf, index_cst(0), subVecOff);
    for (int64_t k = 0; k < mat_k; ++k) {
      Value rhsVec = nextRhsVec;

      // Load next vector in advance to hide load latency.
      if (k != mat_k - 1)
        nextRhsVec = loadRow(loc, rewriter, inputSubVecTy, rhsBuf,
                             index_cst(k + 1), subVecOff);

      Value nextLhsScalar = loadScalar(loc, rewriter, lhsBuf, 0, k);
      for (int64_t m = 0; m < mat_m; ++m) {
        Value lhsScalar = nextLhsScalar;

        // Load next value in advance to hide load latency.
        if (m != mat_m - 1)
          nextLhsScalar = loadScalar(loc, rewriter, lhsBuf, m + 1, k);

        // Call intrinsic to do macc
        Value newAccVec;
        if (k == 0 && isAccZeroInit) {
          auto intrinsicName = getMulIntrinsicName(
              rewriter, "mul", inputElemTy.isFloat(), isWidening);
          SmallVector<Value> args;
          auto poison = rewriter.create<LLVM::PoisonOp>(loc, outputSubVecTy);
          if (inputElemTy.isInteger()) {
            args = {poison, rhsVec, lhsScalar, curVl};
          } else {
            args = {poison, rhsVec, lhsScalar, c_frm_dyn, curVl};
          }
          auto callIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, outputSubVecTy, intrinsicName, args);
          accVecs.push_back(callIntrinsicOp.getResult(0));
        } else {
          auto intrinsicName = getMulIntrinsicName(
              rewriter, "macc", inputElemTy.isFloat(), isWidening);
          SmallVector<Value> args;
          if (inputElemTy.isInteger()) {
            args = {accVecs[m], lhsScalar, rhsVec, curVl, c_tumu};
          } else {
            args = {accVecs[m], lhsScalar, rhsVec, c_frm_dyn, curVl, c_tumu};
          }
          auto callIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
              loc, outputSubVecTy, intrinsicName, args);
          accVecs[m] = callIntrinsicOp.getResult(0);
        }
      }
    }
    storeRows(loc, accBuf, accVecs, subVecOff, rewriter);

  } // end of for-op: rewriter will be set back to where it was automatically

  // The result is in accBuf. We should load it and replace the original
  // constraction result.
  VectorType resTy = outputMatTy.cloneWith(std::nullopt, outputElemTy);
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
  const int64_t baseVlen = 64;
  const int64_t baseInputVl = baseVlen / inputElemBitWidth;

  VectorType outputMatTy = cast<VectorType>(dotOp.getC().getType());
  VectorType inputSubVecTy =
      VectorType::get({baseInputVl}, inputElemTy, {true});
  VectorType outputSubVecTy =
      VectorType::get({baseInputVl}, outputElemTy, {true});

  Value resMat = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(outputMatTy));

  Value c_tumu = int_cst(rewriter.getI64Type(), 3);
  Value c_frm_dyn = int_cst(rewriter.getI64Type(), 7);
  Value cIndex_0 = index_cst(0);
  Value cIndex_1 = index_cst(1);
  Value cIndex_baseInputVl = index_cst(baseInputVl);
  Value cIndex_k = index_cst(mat_k);

  Value vscale = getVscale(loc, rewriter);
  Value vl = op_muli(vscale, cIndex_baseInputVl);
  Value numSubVec = rewriter.create<arith::CeilDivSIOp>(loc, cIndex_k, vl);

  for (int64_t m = 0; m < mat_m; m++) {
    for (int64_t n = 0; n < mat_n; n++) {
      // First round: use vmul.vv to avoid 0 initialize
      Value accInit;
      {
        // Get VL
        Value curVl =
            op_index_cast(rewriter.getI64Type(), op_minui(vl, cIndex_k));
        // Get operands
        Value lhsVec = loadRow(loc, rewriter, inputSubVecTy, lhsBuf,
                               index_cst(m), cIndex_0);
        Value rhsVec = loadCol(loc, rewriter, inputSubVecTy, rhsBuf, cIndex_0,
                               index_cst(n));
        // Prepare and call intrinsic
        auto intrinsicName = getMulIntrinsicName(
            rewriter, "mul", inputElemTy.isFloat(), isWidening);
        SmallVector<Value> args;
        auto poison = rewriter.create<LLVM::PoisonOp>(loc, outputSubVecTy);
        if (inputElemTy.isInteger()) {
          args = {poison, lhsVec, rhsVec, curVl};
        } else {
          args = {poison, lhsVec, rhsVec, c_frm_dyn, curVl};
        }
        auto callIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
            loc, outputSubVecTy, intrinsicName, args);
        // Set intrinsic result as initial accumulator
        accInit = callIntrinsicOp.getResult(0);
      }

      // Following rounds: use vmacc.vv
      auto forOp = rewriter.create<scf::ForOp>(
          loc, cIndex_1, numSubVec, cIndex_1, ArrayRef<Value>{accInit});
      {
        // Get VL
        OpBuilder::InsertionGuard guard(rewriter);
        auto forBody = forOp.getBody();
        rewriter.setInsertionPointToStart(forBody);
        Value iv = forOp.getInductionVar();
        Value subVecOff = op_muli(iv, vl);
        Value remaining = op_subi(cIndex_k, subVecOff);
        Value curVl =
            op_index_cast(rewriter.getI64Type(), op_minui(vl, remaining));
        // Get operands
        Value lhsVec = loadRow(loc, rewriter, inputSubVecTy, lhsBuf,
                               index_cst(m), subVecOff);
        Value rhsVec = loadCol(loc, rewriter, inputSubVecTy, rhsBuf, subVecOff,
                               index_cst(n));
        Value sumVec = forOp.getRegionIterArg(0);
        // Prepare and call intrinsic
        auto intrinsicName = getMulIntrinsicName(
            rewriter, "macc", inputElemTy.isFloat(), isWidening);
        SmallVector<Value> args;
        if (inputElemTy.isInteger()) {
          args = {sumVec, lhsVec, rhsVec, curVl, c_tumu};
        } else {
          args = {sumVec, lhsVec, rhsVec, c_frm_dyn, curVl, c_tumu};
        }
        auto callIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
            loc, outputSubVecTy, intrinsicName, args);
        // Yield intrinsic result
        rewriter.setInsertionPointToEnd(forBody);
        rewriter.create<mlir::scf::YieldOp>(
            loc, ValueRange{callIntrinsicOp.getResult(0)});
      }
      Value sumVec = forOp.getResult(0);

      // Reduction: Sum the accumulated results horizontally.
      Value newAccScalar;
      {
        int64_t baseOutputVl = baseVlen / outputElemTy.getIntOrFloatBitWidth();
        VectorType redVecTy =
            VectorType::get({baseOutputVl}, outputElemTy, {true});
        Value poison = rewriter.create<LLVM::PoisonOp>(loc, redVecTy);
        Value redSumScalar;
        if (isAccZeroInit) {
          auto zeroAttr = rewriter.getZeroAttr(outputElemTy);
          redSumScalar =
              rewriter.create<arith::ConstantOp>(loc, outputElemTy, zeroAttr);
        } else {
          redSumScalar = loadScalar(loc, rewriter, accBuf, m, n);
        }
        Value redSum = rewriter.create<vector::InsertOp>(
            loc, redSumScalar, poison, SmallVector<int64_t>({0}));
        Value curVl =
            op_index_cast(rewriter.getI64Type(), op_minui(vl, cIndex_k));
        SmallVector<Value> args;
        if (outputElemTy.isInteger()) {
          args = {poison, sumVec, redSum, curVl};
        } else {
          args = {poison, sumVec, redSum, c_frm_dyn, curVl};
        }
        auto intrinsicName = getReduceIntrinsicName(
            rewriter, "sum", outputElemTy.isFloat(), "u");
        auto callIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
            loc, redVecTy, intrinsicName, args);
        auto newRedSum = callIntrinsicOp.getResult(0);
        newAccScalar = rewriter.create<vector::ExtractOp>(loc, newRedSum, 0);
      }
      resMat = rewriter.create<vector::InsertOp>(loc, newAccScalar, resMat,
                                                 SmallVector<int64_t>({m, n}));
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

  // Cast input data if required and prepare input buffer. It might be
  // temporary buffers with stored vectors or the original input memory.
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