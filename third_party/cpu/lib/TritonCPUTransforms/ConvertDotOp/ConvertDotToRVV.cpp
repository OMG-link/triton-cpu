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
  uint64_t m, k, n;

  // Memory buffer holding LHS. Can be empty if LHS is not a result of a
  // simple load.
  MemBuffer lhsBuf;
  // Memory buffer holding RHS. Can be empty if RHS is not a result of a
  // simple load.
  MemBuffer rhsBuf;

  // If accumulator is updated in a loop, then this flag indicates if we
  // should keep it in registers the whole loop.
  bool keepAccOnRegs;
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
  candidate.keepAccOnRegs = isLoopCarriedAcc(op.getC());

  // FIXME: For this demo, we simply assume N=32 and VLEN=256
  if (candidate.n != 32)
    return false;

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

Value loadRow(Location loc, VectorType resTy, const MemBuffer &buf, int64_t m,
              PatternRewriter &rewriter) {
  assert(!buf.empty());
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, indices[indices.size() - 2], m, rewriter);
  return rewriter.create<vector::LoadOp>(loc, resTy, buf.memRef, indices);
}

void storeRow(Location loc, const MemBuffer &buf, int64_t rowIdx, Value vec,
              PatternRewriter &rewriter) {
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, buf.indices[indices.size() - 2], rowIdx, rewriter);
  rewriter.create<vector::StoreOp>(loc, vec, buf.memRef, indices);
}

void storeRows(Location loc, const MemBuffer &buf,
               const SmallVector<Value> &vecs, PatternRewriter &rewriter) {
  SmallVector<Value> indices = buf.indices;
  for (int64_t m = 0; m < vecs.size(); ++m)
    storeRow(loc, buf, m, vecs[m], rewriter);
}

SmallVector<Value> extractRows(Location loc, Value vec,
                               PatternRewriter &rewriter) {
  VectorType vecTy = cast<VectorType>(vec.getType());
  SmallVector<Value> res;
  for (int64_t m = 0; m < vecTy.getDimSize(0); ++m) {
    auto row =
        rewriter.create<vector::ExtractOp>(loc, vec, SmallVector<int64_t>({m}));
    res.push_back(row);
  }
  return res;
}

Value mergeRows(Location loc, VectorType resTy, const SmallVector<Value> &tiles,
                PatternRewriter &rewriter) {
  Value res =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resTy));
  for (int64_t m = 0; m < tiles.size(); ++m)
    res = rewriter.create<vector::InsertOp>(loc, tiles[m], res,
                                            SmallVector<int64_t>({m}));
  return res;
}

StringAttr getIntrinsicName(MLIRContext *ctx, bool isInt, bool isWidening) {
  if (isInt) {
    if (isWidening) {
      return StringAttr::get(ctx, "llvm.riscv.vwmacc");
    } else {
      return StringAttr::get(ctx, "llvm.riscv.vmacc");
    }
  } else {
    if (isWidening) {
      return StringAttr::get(ctx, "llvm.riscv.vfwmacc");
    } else {
      return StringAttr::get(ctx, "llvm.riscv.vfmacc");
    }
  }
}

LogicalResult convertRvvCandidate(RvvDotOpCandidate &candidate,
                                  PatternRewriter &rewriter) {
  // FIXME: In practice, we should get VLEN dynamically
  const int64_t VLEN = 256;

  cpu::DotOp op = candidate.op;
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();
  VectorType outputMatrixTy = cast<VectorType>(op.getC().getType());
  int64_t inputElemBitWidth = candidate.inputElemTy.getIntOrFloatBitWidth();
  VectorType inputVecTy =
      VectorType::get({VLEN / 8}, candidate.inputElemTy, {false});
  VectorType outputVecTy =
      VectorType::get({VLEN / 8}, candidate.outputElemTy, {false});

  Value vl =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(32));
  Value tumu =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(3));
  Value frm_dyn =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(7));

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
  Value accToStore = acc;
  scf::ForOp forOp;
  if (candidate.keepAccOnRegs) {
    forOp = cast<scf::ForOp>(op->getParentOp());
    accToStore = getInitAccValue(acc);
  }

  SmallVector<Value> accVecs;
  SmallVector<Value> accInitVecs;
  if (candidate.keepAccOnRegs) {
    // Initial tile values are loaded before the loop and then directly
    // used within the loop. Later, new iter values will be added to
    // add loop carried-dependencies for accumulator tiles and accInitTiles
    // will be used as initializers for them.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forOp);
    LDBG("Loading accumulator to tiles before the loop.");
    accInitVecs = extractRows(loc, accToStore, rewriter);
    accVecs = accInitVecs;
  } else {
    accVecs = extractRows(loc, acc, rewriter);
  }

  Value nextRhsVec = loadRow(loc, inputVecTy, rhsBuf, 0, rewriter);
  for (int64_t k = 0; k < candidate.k; ++k) {
    Value rhsVec = nextRhsVec;

    // Load next vector in advance to hide load latency.
    if (k != candidate.k - 1)
      nextRhsVec = loadRow(loc, inputVecTy, rhsBuf, k + 1, rewriter);

    Value nextLhsScalar = loadScalar(loc, lhsBuf, 0, k, rewriter);
    for (int64_t m = 0; m < candidate.m; ++m) {
      Value lhsScalar = nextLhsScalar;

      // Load next value in advance to hide load latency.
      if (m != candidate.m - 1)
        nextLhsScalar = loadScalar(loc, lhsBuf, m + 1, k, rewriter);

      auto intrinsicName = getIntrinsicName(
          ctx, candidate.inputElemTy.isInteger(), candidate.isWidening);
      SmallVector<Value> args;
      if (candidate.inputElemTy.isInteger()) {
        args = {
            accVecs[m], // Accumulator
            lhsScalar,  // Scalar
            rhsVec,     // Vector
            vl,         // vl
            tumu        // tu, mu
        };
      } else {
        args = {
            accVecs[m], // Accumulator
            lhsScalar,  // Scalar
            rhsVec,     // Vector
            frm_dyn,    // float round mode
            vl,         // vector length
            tumu        // tu, mu
        };
      }
      auto callInstrOp = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, outputVecTy, intrinsicName, args);
      accVecs[m] = callInstrOp.getResult(0);
    }
  }

  if (candidate.keepAccOnRegs) {
    // In this case we have the whole accumulator/result on tiles. Loop
    // carried dependencies are not in place yet and should be added.
    // After the loop, resulting tiles should either be stored to the
    // output buffer, or moved to a vector through a temporary buffer.

    // We don't need the original accumulator and contraction op anymore.
    // Directly yield orig accumulator value, so it would be later removed
    // as unused. The original contraction can be removed right away.
    int64_t origResIdx = op.getResult().getUses().begin()->getOperandNumber();
    rewriter.replaceOp(op, op.getC());

    // Now, replace the loop with a new one to add loop carried dependency for
    // accumulator tiles.
    LDBG("Rewrite loop to introduce loop carried dependencies for accumulator "
         "tiles.");
    SmallVector<Value> newInitOperands;
    SmallVector<Value> newYieldedValues;
    for (int64_t m = 0; m < candidate.m; ++m) {
      LDBG("Initial value\n  " << accInitVecs[m] << "\nis combined with\n  "
                               << accVecs[m]);
      newInitOperands.push_back(accInitVecs[m]);
      newYieldedValues.push_back(accVecs[m]);
    }
    auto newForOp = cast<scf::ForOp>(*forOp.replaceWithAdditionalYields(
        rewriter, newInitOperands, true,
        [&newYieldedValues](OpBuilder &b, Location loc,
                            ArrayRef<BlockArgument> newBBArgs) {
          return newYieldedValues;
        }));

    // The resulting tiles are now in the new loop results.
    auto resVecs = newForOp.getResults().take_back(newYieldedValues.size());
    for (int64_t m = 0; m < candidate.m; ++m)
      accVecs[m] = resVecs[m];

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newForOp);
    // Collect all results into a single vector.
    LDBG("Merging resulting rows to replace loop result.");
    VectorType resTy =
        outputMatrixTy.cloneWith(std::nullopt, candidate.outputElemTy);
    Value newVal = mergeRows(loc, resTy, accVecs, rewriter);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, outputMatrixTy.getElementType(), rewriter);
    rewriter.replaceAllUsesWith(newForOp.getResult(origResIdx), newVal);
  } else {
    // The result is in the buffer. We should load it and replace the original
    // constraction result.
    LDBG("Merging resulting rows to replace orig op result.");
    VectorType resTy =
        outputMatrixTy.cloneWith(std::nullopt, candidate.outputElemTy);
    Value newVal = mergeRows(loc, resTy, accVecs, rewriter);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, outputMatrixTy.getElementType(), rewriter);
    rewriter.replaceOp(op, newVal);
  }

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