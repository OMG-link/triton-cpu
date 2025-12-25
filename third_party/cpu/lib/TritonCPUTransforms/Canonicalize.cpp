#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

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

    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), reshape.getType(), op.getSource(), op.getIndices(),
        *newAffineMap, op.getPadding(), op.getMask(),
        rewriter.getBoolArrayAttr(SmallVector(dstTy.getRank(), true)));
    rewriter.replaceOp(op, newReadOp);

    return success();
  }
};

struct Canonicalize : public triton::cpu::impl::CanonicalizeBase<Canonicalize> {
  Canonicalize() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldReadShapeCast>(context);

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
