#include "Utils.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-cpu-transforms-convert-transpose-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRANSPOSEOP
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

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
      newAffineMapExprs[i] = oldAffineMap.getResult(perm[i]);
    }
    auto newAffineMap = AffineMap::get(
        oldAffineMap.getNumDims(), 0, newAffineMapExprs, rewriter.getContext());

    writeOp.getVectorMutable().set(input);
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
      newAffineMapExprs[perm[i]] = oldAffineMap.getResult(i);
    }
    auto newAffineMap = AffineMap::get(
        oldAffineMap.getNumDims(), 0, newAffineMapExprs, rewriter.getContext());

    // Create new transfer_read
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        oldReadOp.getLoc(), output.getType(), oldReadOp.getSource(),
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

    /// Create a buffer so that we can do transpose through memory
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(transposeOp);
      auto loc = transposeOp.getLoc();
      // FIXME:
      // Can we use a fixed-size buffer to avoid allocate large memory buffer?
      auto memRefTy =
          MemRefType::get(outputTy.getShape(), outputTy.getElementType());
      Value memRef = rewriter.create<memref::AllocaOp>(
          loc, memRefTy, rewriter.getI64IntegerAttr(64));
      LDBG("  Created a buffer for transpose: " << memRef);
      int64_t rank = outputTy.getRank();
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value> indices(rank, zeroIdx);
      auto writeBufferOp = rewriter.create<vector::TransferWriteOp>(
          loc, output, memRef, indices, SmallVector<bool>(rank, true));
      auto readBufferOp = rewriter.create<vector::TransferReadOp>(
          loc, outputTy, memRef, indices, SmallVector<bool>(rank, true));
      output.replaceUsesWithIf(readBufferOp.getResult(),
                               [&writeBufferOp](mlir::OpOperand &oper) -> bool {
                                 return oper.getOwner() != writeBufferOp;
                               });
      return transposeWhenWriting(transposeOp, rewriter);
    }
  }
};

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
  auto memRefTy = dyn_cast<MemRefType>(op.getSource().getType());
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

static Value getLastDimStride(Location loc, PatternRewriter &rewriter,
                              VectorTransferOpInterface op) {
  auto memRef = op.getSource();
  auto memRefTy = cast<MemRefType>(memRef.getType());
  auto memRefMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, memRef);
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
    Value memRef = op.getSource();
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
    Value result = rewriter.create<ub::PoisonOp>(loc, matTy);

    Value vl =
        rewriter.create<arith::ConstantIntOp>(loc, k, rewriter.getI64Type());
    Value stride_index = getLastDimStride(loc, rewriter, op);
    Value stride_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), stride_index);

    // Calculate the starting point of matrix
    Value basePtr = createMemRefToRawPtr(loc, rewriter, memRef);
    auto strides =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, memRef)
            .getStrides();
    Value offset = createI64(loc, rewriter, 0);
    for (int64_t i = 0; i < strides.size(); i++) {
      auto thisOffset = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(),
          rewriter.create<arith::MulIOp>(loc, indices[i], strides[i]));
      offset = rewriter.createOrFold<arith::AddIOp>(loc, offset, thisOffset);
    }
    basePtr = createGep(loc, rewriter, basePtr, matElemTy, offset);

    for (int64_t i_m = 0; i_m < m; i_m += 4) {
      int64_t mr = std::min(m - i_m, int64_t(4));
      Value ptr = createGep(loc, rewriter, basePtr, matElemTy,
                            createI64(loc, rewriter, i_m));
      Value mrTuple = rvv::intrinsic::createLoadStridedSegment(
          loc, rewriter, mr, rowType_scalable, ptr, stride_i64, vl);
      for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
        Value row_scalable =
            rvv::intrinsic::extractFromRvvTuple(loc, rewriter, mrTuple, i_mr);
        Value row_fixed =
            convertToFixedVector(loc, rewriter, row_scalable, rowType_fixed);
        result = rewriter.create<vector::InsertOp>(loc, row_fixed, result,
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
    Value mat = op.getVector();
    VectorType matTy = op.getVector().getType();
    Type matElemTy = matTy.getElementType();
    int64_t m = matTy.getDimSize(0);
    int64_t k = matTy.getDimSize(1);
    Value memRef = op.getSource();
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
        rewriter.create<arith::ConstantIntOp>(loc, k, rewriter.getI64Type());
    Value stride_index = getLastDimStride(loc, rewriter, op);
    Value stride_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), stride_index);

    // Calculate the starting point of matrix
    Value basePtr = createMemRefToRawPtr(loc, rewriter, memRef);
    auto strides =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, memRef)
            .getStrides();
    Value offset = createI64(loc, rewriter, 0);
    for (int64_t i = 0; i < strides.size(); i++) {
      auto thisOffset = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(),
          rewriter.create<arith::MulIOp>(loc, indices[i], strides[i]));
      offset = rewriter.createOrFold<arith::AddIOp>(loc, offset, thisOffset);
    }
    basePtr = createGep(loc, rewriter, basePtr, matElemTy, offset);

    for (int64_t i_m = 0; i_m < m; i_m += 4) {
      int64_t mr = std::min(m - i_m, int64_t(4));
      Type tupleTy =
          rvv::intrinsic::getRvvTupleType(rewriter, rowType_scalable, mr);
      Value matToStore = rewriter.create<ub::PoisonOp>(loc, tupleTy);
      // TODO: We have another intrinsic that can create a whole tuple at a time
      for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
        Value row_fixed =
            rewriter.create<vector::ExtractOp>(loc, mat, i_m + i_mr);
        Value row_scalable =
            convertToScalableVector(loc, rewriter, row_fixed, rowType_scalable);
        matToStore = rvv::intrinsic::insertToRvvTuple(loc, rewriter, matToStore,
                                                      row_scalable, i_mr);
      }
      Value ptr = createGep(loc, rewriter, basePtr, matElemTy,
                            createI64(loc, rewriter, i_m));
      rvv::intrinsic::createStoreStridedSegment(loc, rewriter, matToStore, ptr,
                                                stride_i64, vl);
    }

    rewriter.eraseOp(op);
    LDBG("  Success.");
    return success();
  }
};

struct ConvertTransposeOp
    : public triton::cpu::impl::ConvertTransposeOpBase<ConvertTransposeOp> {
  ConvertTransposeOp() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);

    auto arch = getCpuArch();
    auto cpuFeatures = getCpuFeatures();

    if (arch == "riscv64") {
      if (cpuFeatures.find("v") != cpuFeatures.end()) {
        patterns.add<MergeTransposeIntoTransfer>(context);
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertTransposeOp() {
  return std::make_unique<ConvertTransposeOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
