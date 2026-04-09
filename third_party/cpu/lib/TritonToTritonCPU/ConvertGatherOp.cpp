#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
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
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<ub::UBDialect>();
    addLegalDialect<TritonCPUDialect>();

    addIllegalOp<triton::GatherOp>();
  }
};

struct GatherOpConversion : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value baseVec = rewriter.getRemappedValue(op.getSrc());
    Value indexVec = rewriter.getRemappedValue(op.getIndices());

    // Verify axis attribute
    auto baseShape = dyn_cast<ShapedType>(baseVec.getType());
    if (!baseShape) {
      return op.emitError("expected shaped types for src/indices");
    }
    int64_t baseRank = baseShape.getRank();
    auto axisAttr = op.getAxisAttr();
    if (!axisAttr) {
      return op.emitError("missing axis attribute");
    }
    int64_t axis = axisAttr.getValue().getSExtValue();
    // If axis is not the last dimension, we need to transpose the gather
    // dimension to the last one. For simplicity, we currently just assume the
    // axis is always the last dimension here.
    if (axis != baseRank - 1) {
      return op.emitError(
          "currently only lowering gather when axis == last dimension; please "
          "transpose src/indices so axis is innermost before lowering");
    }

    // Dimensions of 'baseVec' can not be dynamic.
    for (int64_t i = 0; i < baseRank; i++) {
      if (baseShape.isDynamicDim(i)) {
        return op.emitError("dimensions of gather source can not be dynamic");
      }
    }

    auto build = [&rewriter, &loc](auto &build, Value baseVec,
                                   Value indexVec) -> Value {
      VectorType baseTy = cast<VectorType>(baseVec.getType());
      VectorType indexTy = cast<VectorType>(indexVec.getType());
      if (baseTy.getRank() == 1) {
        return RGatherOp::create(rewriter, loc, baseVec, indexVec);
      } else {
        Value result = ub::PoisonOp::create(
            rewriter, loc,
            indexTy.cloneWith(std::nullopt, baseTy.getElementType()));
        for (int64_t i = 0; i < baseTy.getDimSize(0); i++) {
          Value subBaseVec =
              vector::ExtractOp::create(rewriter, loc, baseVec, i);
          Value subIndexVec =
              vector::ExtractOp::create(rewriter, loc, indexVec, i);
          Value subResult = build(build, subBaseVec, subIndexVec);
          result =
              vector::InsertOp::create(rewriter, loc, subResult, result, i);
        }
        return result;
      }
    };
    rewriter.replaceOp(op, build(build, baseVec, indexVec));

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
