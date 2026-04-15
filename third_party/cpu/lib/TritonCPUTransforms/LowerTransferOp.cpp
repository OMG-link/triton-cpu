#include "Utils.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-cpu-transforms-lower-transfer-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_LOWERTRANSFEROP
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

bool isTransposedTransfer(VectorTransferOpInterface op) {
  bool isOutOfBounds = op.hasOutOfBoundsDim();
  bool isMasked = bool(op.getMask());
  if (isOutOfBounds || isMasked) {
    // In fact, we can handle such cases.
    // But skip it for simplicity for now.
    LDBG("  Skipped: 'isOutOfBounds || isMasked'");
    return false;
  }
  int64_t rank = op.getVectorType().getRank();
  if (rank != 2) {
    LDBG("  Skipped: 'rank != 2'");
    return false;
  }
  if (op.getVectorType().getDimSize(0) <= 1 &&
      op.getVectorType().getScalableDims()[0] == false) {
    LDBG("  Skipped: size of first dimension should be at least 2");
    return false;
  }
  auto memRefTy = dyn_cast<MemRefType>(op.getBase().getType());
  if (!memRefTy) {
    LDBG("  Skipped: '!memRefTy'");
    return false;
  }
  auto strides = memRefTy.getStridesAndOffset().first;
  auto affineMap = op.getPermutationMap();
  int64_t dim0, dim1;
  if (auto affineDimExpr = dyn_cast<AffineDimExpr>(affineMap.getResult(0))) {
    dim0 = affineDimExpr.getPosition();
  } else {
    LDBG("  Skipped: unknown/invalid type of affine map");
    return false;
  }
  if (auto affineDimExpr = dyn_cast<AffineDimExpr>(affineMap.getResult(1))) {
    dim1 = affineDimExpr.getPosition();
  } else {
    LDBG("  Skipped: unknown/invalid type of affine map");
    return false;
  }
  bool isTransposed = (strides[dim0] == 1 && strides[dim1] != 1);
  if (!isTransposed) {
    LDBG("  Skipped: not transposed");
    return false;
  }
  LDBG("  Pre-check passed");
  return true;
}

static Value getLastDimStride(PatternRewriter &rewriter, Location loc,
                              VectorTransferOpInterface op) {
  auto memRef = op.getBase();
  auto memRefTy = cast<MemRefType>(memRef.getType());
  auto memRefMetadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memRef);
  auto affineMap = op.getPermutationMap();
  assert(affineMap.getNumResults() == 2);
  auto lastDim = cast<AffineDimExpr>(affineMap.getResult(1)).getPosition();
  return memRefMetadata.getStrides()[lastDim];
}

struct LowerTransposedTransferReadOpToRvv
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    LDBG("Lowering to RVV intrinsic: " << op);
    if (!isTransposedTransfer(op))
      return failure();
    Location loc = op.getLoc();
    VectorType matTy = op.getResult().getType();
    Type matElemTy = matTy.getElementType();
    int64_t m = matTy.getDimSize(0);
    int64_t k = matTy.getDimSize(1);
    Value memRef = op.getBase();
    auto indices = op.getIndices();
    int64_t vlen = rvv::getMinimumVlen();

    if (matTy.isScalable()) {
      LDBG("  Failed: scalable dim are not allowed");
      return failure();
    }
    if (std::min(m, int64_t(4)) * k * matElemTy.getIntOrFloatBitWidth() >
        8 * vlen) {
      // FIXME: we can work around by spliting vectors
      LDBG("  Failed: row too long");
      return failure();
    }

    VectorType rowType_fixed = VectorType::get({k}, matElemTy);
    VectorType rowType_scalable;
    if (auto r = rvv::getSmallestScalableTypeThatHolds(rowType_fixed);
        failed(r)) {
      LDBG("  Failed: failed to find suitable scalable vector type");
      return failure();
    } else {
      rowType_scalable = *r;
    }
    Value result = ub::PoisonOp::create(rewriter, loc, matTy);

    Value vl =
        arith::ConstantIntOp::create(rewriter, loc, rewriter.getI64Type(), k);
    Value stride_index = getLastDimStride(rewriter, loc, op);
    Value stride_i64 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), stride_index);

    // Calculate the starting point of matrix
    Value basePtr = createMemRefToRawPtr(rewriter, loc, memRef);
    auto strides =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, memRef)
            .getStrides();
    Value offset = createI64(rewriter, loc, 0);
    for (int64_t i = 0; i < strides.size(); i++) {
      auto thisOffset = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(),
          arith::MulIOp::create(rewriter, loc, indices[i], strides[i]));
      offset = rewriter.createOrFold<arith::AddIOp>(loc, offset, thisOffset);
    }
    basePtr = createGep(rewriter, loc, basePtr, matElemTy, offset);

    for (int64_t i_m = 0; i_m < m; i_m += 4) {
      int64_t mr = std::min(m - i_m, int64_t(4));
      Value ptr = createGep(rewriter, loc, basePtr, matElemTy,
                            createI64(rewriter, loc, i_m));
      Value mrTuple = rvv::intrinsic::createLoadStridedSegment(
          rewriter, loc, mr, rowType_scalable, ptr, stride_i64, vl);
      for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
        Value row_scalable =
            rvv::intrinsic::extractFromRvvTuple(rewriter, loc, mrTuple, i_mr);
        Value row_fixed =
            convertToFixedVector(rewriter, loc, row_scalable, rowType_fixed);
        result = vector::InsertOp::create(rewriter, loc, row_fixed, result,
                                          i_m + i_mr);
      }
    }

    rewriter.replaceOp(op, result);
    LDBG("  Success.");
    return success();
  }
};

struct LowerTransposedTransferWriteOpToRvv
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    LDBG("Lowering to RVV intrinsic: " << op);
    if (!isTransposedTransfer(op))
      return failure();
    Location loc = op.getLoc();
    Value mat = op.getValueToStore();
    VectorType matTy = op.getValueToStore().getType();
    Type matElemTy = matTy.getElementType();
    int64_t m = matTy.getDimSize(0);
    int64_t k = matTy.getDimSize(1);
    Value memRef = op.getBase();
    auto indices = op.getIndices();
    int64_t vlen = rvv::getMinimumVlen();

    if (matTy.isScalable()) {
      LDBG("  Failed: scalable dim are not allowed");
      return failure();
    }
    if (std::min(m, int64_t(4)) * k * matTy.getElementTypeBitWidth() >
        8 * vlen) {
      // FIXME: we can work around by spliting vectors
      LDBG("  Failed: row too long");
      return failure();
    }

    VectorType rowType_fixed = VectorType::get({k}, matElemTy);
    VectorType rowType_scalable;
    if (auto r = rvv::getSmallestScalableTypeThatHolds(rowType_fixed);
        failed(r)) {
      LDBG("  Failed: failed to find suitable scalable vector type");
      return failure();
    } else {
      rowType_scalable = *r;
    }

    Value vl =
        arith::ConstantIntOp::create(rewriter, loc, rewriter.getI64Type(), k);
    Value stride_index = getLastDimStride(rewriter, loc, op);
    Value stride_i64 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), stride_index);

    // Calculate the starting point of matrix
    Value basePtr = createMemRefToRawPtr(rewriter, loc, memRef);
    auto strides =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, memRef)
            .getStrides();
    Value offset = createI64(rewriter, loc, 0);
    for (int64_t i = 0; i < strides.size(); i++) {
      auto thisOffset = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(),
          arith::MulIOp::create(rewriter, loc, indices[i], strides[i]));
      offset = rewriter.createOrFold<arith::AddIOp>(loc, offset, thisOffset);
    }
    basePtr = createGep(rewriter, loc, basePtr, matElemTy, offset);

    for (int64_t i_m = 0; i_m < m; i_m += 4) {
      int64_t mr = std::min(m - i_m, int64_t(4));
      Type tupleTy =
          rvv::intrinsic::getRvvTupleType(rewriter, rowType_scalable, mr);
      Value matToStore = ub::PoisonOp::create(rewriter, loc, tupleTy);
      // TODO: We have another intrinsic that can create a whole tuple at a time
      for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
        Value row_fixed =
            vector::ExtractOp::create(rewriter, loc, mat, i_m + i_mr);
        Value row_scalable =
            convertToScalableVector(rewriter, loc, row_fixed, rowType_scalable);
        matToStore = rvv::intrinsic::insertToRvvTuple(rewriter, loc, matToStore,
                                                      row_scalable, i_mr);
      }
      Value ptr = createGep(rewriter, loc, basePtr, matElemTy,
                            createI64(rewriter, loc, i_m));
      rvv::intrinsic::createStoreStridedSegment(rewriter, loc, matToStore, ptr,
                                                stride_i64, vl);
    }

    rewriter.eraseOp(op);
    LDBG("  Success.");
    return success();
  }
};

struct LowerTransferOp
    : public triton::cpu::impl::LowerTransferOpBase<LowerTransferOp> {
  LowerTransferOp() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);

    auto arch = getCpuArch();
    auto cpuFeatures = getCpuFeatures();

    if (arch == "riscv64") {
      if (cpuFeatures.find("v") != cpuFeatures.end()) {
        patterns.add<LowerTransposedTransferReadOpToRvv>(context);
        patterns.add<LowerTransposedTransferWriteOpToRvv>(context);
      }
    }

    if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createLowerTransferOp() {
  return std::make_unique<LowerTransferOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
