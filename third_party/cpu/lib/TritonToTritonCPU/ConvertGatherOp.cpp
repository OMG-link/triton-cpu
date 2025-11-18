#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTGATHEROP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class GatherConversionTarget : public ConversionTarget {
public:
  explicit GatherConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<TritonCPUDialect>();

    addIllegalOp<triton::GatherOp>();
  }
};

struct GatherOpConversion : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Converting " << op << '\n';
    Location loc = op.getLoc();

    // Remapped operands (after other rewrites / mapping)
    Value baseVec = rewriter.getRemappedValue(op.getSrc());
    Value indexVec = rewriter.getRemappedValue(op.getIndices());

    // Convert the result type (tensor -> vector) via the type converter.
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    auto vecResultType = dyn_cast<VectorType>(convertedType);
    if (!vecResultType) {
      return op.emitError("converted result type is not a VectorType");
    }

    // Defensive: shaped inputs
    auto baseShape = dyn_cast<ShapedType>(baseVec.getType());
    auto indexVecShape = dyn_cast<ShapedType>(indexVec.getType());
    if (!baseShape || !indexVecShape) {
      return op.emitError("expected shaped types for src/indices");
    }

    int64_t baseRank = baseShape.getRank();
    int64_t indexVecRank = indexVecShape.getRank();
    int64_t resRank = vecResultType.getRank();

    // Triton ODS said src, indices, result have same rank. Check consistency.
    // FIXME: This seems weird. Need more tests to verify.
    if (!(baseRank == indexVecRank && indexVecRank == resRank)) {
      return op.emitError("src, indices and result must have same rank");
    }

    // Axis attribute
    auto axisAttr = op.getAxisAttr();
    if (!axisAttr) {
      return op.emitError("missing axis attribute");
    }
    int64_t axis = axisAttr.getValue().getSExtValue();
    if (axis < 0 || axis >= baseRank) {
      return op.emitError("axis attribute out of range");
    }

    // For correctness with vector.gather (which only varies index in innermost
    // dimension), require axis to be the innermost dimension for the direct
    // lowering. If it's not, the transformation must transpose/permute src and
    // indices so that axis becomes the last dimension. For simplicity, we just
    // assume the axis is always the last dimension here.
    if (axis != baseRank - 1) {
      return op.emitError(
          "currently only lowering gather when axis == last dimension; please "
          "transpose src/indices so axis is innermost before lowering");
    }
    // -------------------------
    // Prepare base: base must be a memref or ranked tensor
    // -------------------------
    VectorType baseVecType = dyn_cast<VectorType>(baseVec.getType());
    if (!baseVecType) {
      return op.emitError("expected vector type for src");
    }
    MemRefType baseMemRefType =
        MemRefType::get(baseVecType.getShape(), baseVecType.getElementType());
    Value baseMemRef = rewriter.create<memref::AllocOp>(loc, baseMemRefType);
    auto transferWriteIndices = SmallVector<Value>(
        baseRank, rewriter.create<arith::ConstantIndexOp>(loc, 0));
    rewriter.create<vector::TransferWriteOp>(loc, baseVec, baseMemRef,
                                             transferWriteIndices);

    // -------------------------
    // Prepare indices
    // -------------------------
    SmallVector<Value> indices(baseRank,
                               rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // -------------------------
    // Build mask: vector of i1, same shape as result, filled with true.
    // -------------------------
    VectorType maskType =
        VectorType::get(vecResultType.getShape(), rewriter.getI1Type());
    DenseElementsAttr maskAttr =
        DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true));
    Value mask =
        rewriter.create<mlir::arith::ConstantOp>(loc, maskType, maskAttr);

    // -------------------------
    // Build pass_thru (poison of result vector type)
    // -------------------------
    Value passThru = rewriter.create<mlir::LLVM::PoisonOp>(loc, vecResultType);

    // -------------------------
    // Create the vector.gather op:
    // build(loc, resultType, base, indices(ValueRange offsets), index_vec,
    // mask, pass_thru)
    // -------------------------
    auto retOp = rewriter.replaceOpWithNewOp<vector::GatherOp>(
        op, vecResultType, baseMemRef, indices, indexVec, mask, passThru);
    llvm::errs() << "-> " << retOp << '\n';
    return success();
  }
};

struct ConvertGatherOp
    : public triton::impl::ConvertGatherOpBase<ConvertGatherOp> {
  using ConvertGatherOpBase::ConvertGatherOpBase;

  ConvertGatherOp() : ConvertGatherOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    GatherConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<GatherOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertGatherOp() {
  return std::make_unique<ConvertGatherOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
