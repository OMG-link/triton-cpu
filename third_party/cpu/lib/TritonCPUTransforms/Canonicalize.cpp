#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-cpu-transforms-canonicalize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CANONICALIZE
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Suppose srcTy matches the output of oldAffineMap, and will be reshaped to
// dstTy. This function returns a new affine map that produces dstTy directly.
static FailureOr<AffineMap> getAffineMapAfterReshape(MLIRContext *C,
                                                     AffineMap oldAffineMap,
                                                     VectorType srcTy,
                                                     VectorType dstTy) {
  SmallVector<AffineExpr> newAffineMapExprs(dstTy.getRank(),
                                            getAffineConstantExpr(0, C));
  int64_t i_srcTy = 0;
  int64_t i_dstTy = 0;
  for (; i_srcTy < srcTy.getRank(); i_srcTy++) {
    if (srcTy.getDimSize(i_srcTy) != 1) {
      while (true) {
        if (i_dstTy >= dstTy.getRank())
          return failure();
        if (dstTy.getDimSize(i_dstTy) != 1)
          break;
        i_dstTy++;
      }
      if (srcTy.getDimSize(i_srcTy) != dstTy.getDimSize(i_dstTy))
        return failure();
      newAffineMapExprs[i_dstTy] = oldAffineMap.getResult(i_srcTy);
      i_dstTy++;
    }
  }
  return AffineMap::get(oldAffineMap.getNumDims(), 0, newAffineMapExprs, C);
}

// Fold transfer write and the input shape cast that removes/inserts
// dimensions with size 1.
struct FoldReadShapeCast : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return failure();

    if (op.isMasked())
      return failure();

    if (op.hasOutOfBoundsDim())
      return failure();

    auto reshape = dyn_cast<vector::ShapeCastOp>(*op->user_begin());
    if (!reshape)
      return failure();

    VectorType srcTy = reshape.getSourceVectorType();
    VectorType dstTy = reshape.getResultVectorType();
    auto oldPermMap = op.getPermutationMap();
    auto newAffineMap = getAffineMapAfterReshape(rewriter.getContext(),
                                                 oldPermMap, srcTy, dstTy);
    if (failed(newAffineMap))
      return LogicalResult(newAffineMap);

    auto newReadOp = vector::TransferReadOp::create(
        rewriter, op.getLoc(), reshape.getType(), op.getBase(), op.getIndices(),
        *newAffineMap, op.getPadding(), op.getMask(),
        rewriter.getBoolArrayAttr(SmallVector(dstTy.getRank(), true)));
    rewriter.replaceOp(op, newReadOp);

    return success();
  }
};

struct MergeTransposeIntoTransfer
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult transposeWhenWriting(vector::TransposeOp transposeOp,
                                     PatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard guard(rewriter);

    Value input = transposeOp.getOperand();
    Value output = transposeOp.getResult();
    assert(output.hasOneUse());
    auto writeOp = cast<vector::TransferWriteOp>(*output.user_begin());
    LDBG("  Merging transpose into: " << writeOp);

    // Make sure transfer_write has no complex attributes
    if (writeOp.isMasked()) {
      LDBG("  Failed: transfer_write is masked");
      return failure();
    }
    if (writeOp.hasOutOfBoundsDim()) {
      // FIXME: we actually can process this
      LDBG("  Failed: transfer_write has out of bounds dims");
      return failure();
    }

    int64_t rank = writeOp.getVectorType().getRank();
    auto perm = transposeOp.getPermutation();
    assert(rank == perm.size());

    // Permute transfer_write.affine_map
    auto oldAffineMap = writeOp.getPermutationMap();
    assert(rank == oldAffineMap.getNumResults());
    SmallVector<AffineExpr> newAffineMapExprs(rank);
    for (int64_t i = 0; i < rank; i++) {
      newAffineMapExprs[perm[i]] = oldAffineMap.getResult(i);
    }
    auto newAffineMap = AffineMap::get(
        oldAffineMap.getNumDims(), 0, newAffineMapExprs, rewriter.getContext());

    writeOp.getValueToStoreMutable().set(input);
    writeOp.setPermutationMap(newAffineMap);

    LDBG("  Success, new transfer_write: " << writeOp);
    return success();
  }

  LogicalResult transposeWhenReading(vector::TransposeOp transposeOp,
                                     PatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard guard(rewriter);

    Value input = transposeOp.getOperand();
    Value output = transposeOp.getResult();
    auto oldReadOp = cast<vector::TransferReadOp>(input.getDefiningOp());
    LDBG("  Merging transpose into: " << oldReadOp);

    // Make sure transfer_read has no complex attributes
    if (oldReadOp.isMasked()) {
      LDBG("  Failed: transfer_read is masked");
      return failure();
    }
    if (oldReadOp.hasOutOfBoundsDim()) {
      // FIXME: we actually can process this
      LDBG("  Failed: transfer_read has out of bounds dims");
      return failure();
    }

    int64_t rank = oldReadOp.getResult().getType().getRank();
    auto perm = transposeOp.getPermutation();
    assert(rank == perm.size());

    // Permute transfer_read.affine_map
    auto oldAffineMap = oldReadOp.getPermutationMap();
    assert(rank == oldAffineMap.getNumResults());
    SmallVector<AffineExpr> newAffineMapExprs(rank);
    for (int64_t i = 0; i < rank; i++) {
      newAffineMapExprs[i] = oldAffineMap.getResult(perm[i]);
    }
    auto newAffineMap = AffineMap::get(
        oldAffineMap.getNumDims(), 0, newAffineMapExprs, rewriter.getContext());

    // Create new transfer_read
    auto newReadOp = vector::TransferReadOp::create(
        rewriter, oldReadOp.getLoc(), output.getType(), oldReadOp.getBase(),
        oldReadOp.getIndices(), newAffineMap, oldReadOp.getPadding(),
        oldReadOp.getMask(), oldReadOp.getInBounds());
    rewriter.replaceOp(transposeOp, newReadOp);

    LDBG("  Success, new transfer_read: " << newReadOp);
    return success();
  }

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Attempt to merge transpose with memory access: " << transposeOp);

    Value input = transposeOp.getOperand();
    VectorType inputTy = cast<VectorType>(input.getType());
    Value output = transposeOp.getResult();
    VectorType outputTy = cast<VectorType>(output.getType());

    /// Try to merge transpose with transfer_write
    // If transpose has multiple uses rather than a single transfer_write, it
    // cannot be safely merged into a single transfer_write. Fallback to other
    // strategies in such cases.
    if (output.hasOneUse()) {
      if (auto writeOp =
              dyn_cast<vector::TransferWriteOp>(*output.user_begin())) {
        return transposeWhenWriting(transposeOp, rewriter);
      }
    }

    /// Try to merge transpose with transfer_read
    // 'transposeWhenReading' creates a new transfer_read which will be used by
    // users of transpose. Therefore, single use check is not required.
    if (auto readOp = dyn_cast<vector::TransferReadOp>(input.getDefiningOp());
        readOp) {
      return transposeWhenReading(transposeOp, rewriter);
    }

    return failure();
  }
};

struct Canonicalize : public triton::cpu::impl::CanonicalizeBase<Canonicalize> {
  Canonicalize() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldReadShapeCast>(context);
    patterns.add<MergeTransposeIntoTransfer>(context);

    if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalize() {
  return std::make_unique<Canonicalize>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
