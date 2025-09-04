#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

  candidate.lhsBuf = findInputBuffer(op.getA(), true);
  candidate.rhsBuf = findInputBuffer(op.getB(), false);

  return true;
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

Value loadScalar(Location loc, const MemBuffer &buf, int64_t m, int64_t n,
                 PatternRewriter &rewriter) {
  SmallVector<Value> indices = shiftIndices(loc, buf, m, n, rewriter);
  return rewriter.create<memref::LoadOp>(loc, buf.memRef, indices);
}

Value loadRow(Location loc, VectorType resTy, const MemBuffer &buf,
              int64_t rowOff, const Value &subVecOff,
              PatternRewriter &rewriter) {
  assert(!buf.empty());
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, indices[indices.size() - 2], rowOff, rewriter);
  indices[indices.size() - 1] = subVecOff;
  return rewriter.create<vector::LoadOp>(loc, resTy, buf.memRef, indices);
}

SmallVector<Value> loadRows(Location loc, VectorType rowTy, int64_t rowNum,
                            const MemBuffer &buf, int64_t rowOff,
                            const Value &subVecOff, PatternRewriter &rewriter) {
  SmallVector<Value> vecs;
  vecs.reserve(rowNum);
  for (int64_t m = 0; m < rowNum; ++m)
    vecs.push_back(loadRow(loc, rowTy, buf, rowOff, subVecOff, rewriter));
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

StringAttr getIntrinsicName(PatternRewriter &rewriter, bool isInt,
                            bool isWidening) {
  if (isInt) {
    if (isWidening) {
      return rewriter.getStringAttr("llvm.riscv.vwmacc");
    } else {
      return rewriter.getStringAttr("llvm.riscv.vmacc");
    }
  } else {
    if (isWidening) {
      return rewriter.getStringAttr("llvm.riscv.vfwmacc");
    } else {
      return rewriter.getStringAttr("llvm.riscv.vfmacc");
    }
  }
}

LogicalResult convertRvvCandidate(RvvDotOpCandidate &candidate,
                                  PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  VectorType outputMatTy = cast<VectorType>(op.getC().getType());
  int64_t inputElemBitWidth = candidate.inputElemTy.getIntOrFloatBitWidth();

  const int baseVlen = 64;
  const int64_t baseVl = baseVlen / inputElemBitWidth;

  Value c_tumu = int_cst(rewriter.getI64Type(), 3);
  Value c_frm_dyn = int_cst(rewriter.getI64Type(), 7);
  Value cIndex_0 = index_cst(0);
  Value cIndex_1 = index_cst(1);
  Value c_baseVl = index_cst(baseVl);

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Cast input data if required and prepare input buffer. It might be
  // temporary buffers with stored vectors or the original input memory.
  MemBuffer lhsBuf = candidate.lhsBuf;
  if (lhsBuf.empty()) {
    Value lhs = op.getA();
    lhsBuf = storeToTmpBuffer(loc, lhs, allocaPoint, rewriter);
  }

  MemBuffer rhsBuf = candidate.rhsBuf;
  if (rhsBuf.empty()) {
    Value rhs = op.getB();
    rhsBuf = storeToTmpBuffer(loc, rhs, allocaPoint, rewriter);
  }

  Value acc = op.getC();
  MemBuffer accBuf = storeToTmpBuffer(loc, acc, allocaPoint, rewriter);

  // We will do GEMM by multiplying an <M x 1> vector with an <1 x N> vector.
  // To do so, we multiply <1 x 1> scalar with <1 x N> vector.
  // As N is given by the user, it can be too large for single vector register.
  // We need to divide <1 x N> vector into <1 x VL> one, where VL is the number
  // of elements single vector register can hold.

  // Divide <1 x N> vector into <1 x VL> one.
  // We do this first to achieve the best performance.
  Value vscale =
      rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
  Value vl = op_muli(vscale, c_baseVl);

  Value nVal = index_cst(candidate.n);
  Value numBlocks = rewriter.create<arith::CeilDivSIOp>(loc, nVal, vl);
  auto forOp = rewriter.create<scf::ForOp>(loc, cIndex_0, numBlocks, cIndex_1);
  VectorType inputSubVecTy =
      VectorType::get({baseVl}, candidate.inputElemTy, {true});
  VectorType outputSubVecTy =
      VectorType::get({baseVl}, candidate.outputElemTy, {true});

  // For-op body: this for-op divide <* x N> vector into <* x VL> one
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    // curVl = min(vl, n - iv * vl)
    Value iv = forOp.getInductionVar();
    Value subVecOff = op_muli(iv, vl);
    Value remaining = op_subi(nVal, subVecOff);
    Value curVl = op_index_cast(rewriter.getI64Type(), op_minui(vl, remaining));

    SmallVector<Value> accVecs = loadRows(loc, outputSubVecTy, candidate.m,
                                          accBuf, 0, subVecOff, rewriter);

    Value nextRhsVec =
        loadRow(loc, inputSubVecTy, rhsBuf, 0, subVecOff, rewriter);
    for (int64_t k = 0; k < candidate.k; ++k) {
      Value rhsVec = nextRhsVec;

      // Load next vector in advance to hide load latency.
      if (k != candidate.k - 1)
        nextRhsVec =
            loadRow(loc, inputSubVecTy, rhsBuf, k + 1, subVecOff, rewriter);

      Value nextLhsScalar = loadScalar(loc, lhsBuf, 0, k, rewriter);
      for (int64_t m = 0; m < candidate.m; ++m) {
        Value lhsScalar = nextLhsScalar;

        // Load next value in advance to hide load latency.
        if (m != candidate.m - 1)
          nextLhsScalar = loadScalar(loc, lhsBuf, m + 1, k, rewriter);

        // Call intrinsic to do macc
        auto intrinsicName = getIntrinsicName(
            rewriter, candidate.inputElemTy.isInteger(), candidate.isWidening);
        SmallVector<Value> args;
        if (candidate.inputElemTy.isInteger()) {
          args = {accVecs[m], lhsScalar, rhsVec, curVl, c_tumu};
        } else {
          args = {accVecs[m], lhsScalar, rhsVec, c_frm_dyn, curVl, c_tumu};
        }
        auto callInstrOp = rewriter.create<LLVM::CallIntrinsicOp>(
            loc, accVecs[m].getType(), intrinsicName, args);
        Value newAccVec = callInstrOp.getResult(0);

        // Update accVecs
        accVecs[m] = newAccVec;
      }
    }
    storeRows(loc, accBuf, accVecs, subVecOff, rewriter);

  } // end of for-op: rewriter will be set back to where it was automatically

  // The result is in accBuf. We should load it and replace the original
  // constraction result.
  VectorType resTy =
      outputMatTy.cloneWith(std::nullopt, candidate.outputElemTy);
  Value newAccMat = op_read(outputMatTy, accBuf.memRef, accBuf.indices);
  rewriter.replaceOp(op, newAccMat);

  return success();
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