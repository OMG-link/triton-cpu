#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
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

// Iterates over all matching pairs of “non-unit dimensions” in srcTy and dstTy.
// - Dimensions of size 1 in srcTy are handled separately via onSrcUnitDim.
// - Dimensions of size 1 in dstTy are skipped automatically.
// - When a pair of non-unit dimensions is found, their sizes must match,
//   and the pair is passed to onMatch for processing.
// Returns failure() if any mismatch is encountered.
template <typename OnMatch, typename OnSrcUnitDim>
static LogicalResult walkReshapeDims(VectorType srcTy, VectorType dstTy,
                                     OnMatch &&onMatch,
                                     OnSrcUnitDim &&onSrcUnitDim) {
  int64_t iDstTy = 0;

  for (int64_t iSrcTy = 0; iSrcTy < srcTy.getRank(); ++iSrcTy) {
    if (srcTy.getDimSize(iSrcTy) == 1) {
      if (failed(onSrcUnitDim(iSrcTy)))
        return failure();
      continue;
    }

    while (true) {
      if (iDstTy >= dstTy.getRank())
        return failure();
      if (dstTy.getDimSize(iDstTy) != 1)
        break;
      ++iDstTy;
    }

    if (srcTy.getDimSize(iSrcTy) != dstTy.getDimSize(iDstTy))
      return failure();

    if (failed(onMatch(iSrcTy, iDstTy)))
      return failure();

    ++iDstTy;
  }

  return success();
}

// Suppose srcTy matches the output of oldAffineMap, and will be reshaped to
// dstTy. This function returns a new affine map that produces dstTy directly.
static FailureOr<AffineMap> getAffineMapAfterReshape(MLIRContext *C,
                                                     AffineMap srcAffineMap,
                                                     VectorType srcTy,
                                                     VectorType dstTy) {
  SmallVector<AffineExpr> dstAffineMapExprs(dstTy.getRank(),
                                            getAffineConstantExpr(0, C));

  if (failed(walkReshapeDims(
          srcTy, dstTy,
          [&](int64_t iSrcTy, int64_t iDstTy) {
            dstAffineMapExprs[iDstTy] = srcAffineMap.getResult(iSrcTy);
            return success();
          },
          [&](int64_t) { return success(); })))
    return failure();

  return AffineMap::get(srcAffineMap.getNumDims(), 0, dstAffineMapExprs, C);
}

// Suppose the `in_bounds` attribute of srcTy is oldInBoundsAttr, and srcTy
// will be reshaped to dstTy. This function returns the `in_bounds` attribute
// of dstTy.
static FailureOr<ArrayAttr> getInBoundsAfterReshape(MLIRContext *C,
                                                    ArrayAttr srcInBoundsAttr,
                                                    VectorType srcTy,
                                                    VectorType dstTy) {
  SmallVector<bool> srcInBounds = llvm::map_to_vector(
      srcInBoundsAttr.getAsRange<BoolAttr>(),
      [](BoolAttr attr) -> bool { return attr.getValue(); });

  SmallVector<bool> dstInBounds(dstTy.getRank(), true);

  if (failed(walkReshapeDims(
          srcTy, dstTy,
          [&](int64_t iSrcTy, int64_t iDstTy) {
            dstInBounds[iDstTy] = srcInBounds[iSrcTy];
            return success();
          },
          [&](int64_t iSrcTy) {
            // Dimension of length 1 must always be in bounds.
            return srcInBounds[iSrcTy] ? success() : failure();
          })))
    return failure();

  return ArrayAttr::get(
      C, llvm::map_to_vector(dstInBounds, [C](bool b) -> Attribute {
        return BoolAttr::get(C, b);
      }));
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

    auto reshape = dyn_cast<vector::ShapeCastOp>(*op->user_begin());
    if (!reshape)
      return failure();

    VectorType srcTy = reshape.getSourceVectorType();
    VectorType dstTy = reshape.getResultVectorType();
    auto dstAffineMap = getAffineMapAfterReshape(
        rewriter.getContext(), op.getPermutationMap(), srcTy, dstTy);
    if (failed(dstAffineMap))
      return LogicalResult(dstAffineMap);
    auto dstInBoundsAttr = getInBoundsAfterReshape(
        rewriter.getContext(), op.getInBoundsAttr(), srcTy, dstTy);
    if (failed(dstInBoundsAttr))
      return LogicalResult(dstInBoundsAttr);

    auto newReadOp = vector::TransferReadOp::create(
        rewriter, op.getLoc(), reshape.getType(), op.getBase(), op.getIndices(),
        *dstAffineMap, op.getPadding(), op.getMask(), *dstInBoundsAttr);
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
