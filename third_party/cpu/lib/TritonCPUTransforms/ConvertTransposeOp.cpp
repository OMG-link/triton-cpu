#include "Utils.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

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

LogicalResult transposeWhenWriting(vector::TransposeOp op,
                                   PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  Value input = op.getOperand();
  Value output = op.getResult();
  assert(output.hasOneUse());
  auto writeOp = cast<vector::TransferWriteOp>(*output.user_begin());
  LDBG("  Merging transpose into: " << writeOp);

  // Get MemRef
  auto memRef = writeOp.getSource();
  auto memRefTy = dyn_cast<MemRefType>(memRef.getType());
  if (!memRefTy) {
    LDBG("  Failed: transfer_write did not write into a memref");
    return failure();
  }

  // Make sure transfer_write has no complex attributes
  if (!writeOp.getPermutationMapAttr().isIdentity()) {
    LDBG("  Failed: transfer_write contains complex permutation_map");
    return failure();
  }
  if (writeOp.isMasked()) {
    LDBG("  Failed: transfer_write is masked");
    return failure();
  }
  if (writeOp.hasOutOfBoundsDim()) {
    LDBG("  Failed: transfer_write has out of bounds dims");
    return failure();
  }

  // Convert DenseI64MapAttr to AffineMapAttr for memref::TransposeOp
  auto perm_I64Array = op.getPermutation();
  auto perm_AffineMap =
      AffineMap::getPermutationMap(perm_I64Array, rewriter.getContext());
  auto perm_AffineMapAttr = AffineMapAttr::get(perm_AffineMap);

  // Apply transpose to indices
  auto indices_old = writeOp.getIndices();
  SmallVector<Value> indices_new(indices_old.size());
  for (int64_t i = 0; i < perm_I64Array.size(); i++) {
    indices_new[perm_I64Array[i]] = indices_old[i];
  }

  // Transpose memref
  rewriter.setInsertionPointAfterValue(memRef);
  auto transposedMemRef = rewriter.create<memref::TransposeOp>(
      op.getLoc(), memRef, perm_AffineMapAttr);
  writeOp.getVectorMutable().set(input);
  writeOp.getSourceMutable().set(transposedMemRef);
  for (int64_t i = 0; i < indices_new.size(); i++) {
    writeOp.getIndicesMutable()[i].set(indices_new[i]);
  }

  LDBG("  Success, new transfer_write: " << writeOp);
  return success();
}

LogicalResult transposeWhenReading(vector::TransposeOp op,
                                   PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  Value input = op.getOperand();
  Value output = op.getResult();
  int64_t rank = op.getResultVectorType().getRank();
  auto oldReadOp = cast<vector::TransferReadOp>(input.getDefiningOp());
  LDBG("  Merging transpose into: " << oldReadOp);

  // Get MemRef
  auto memRef = oldReadOp.getSource();
  auto memRefTy = dyn_cast<MemRefType>(memRef.getType());
  if (!memRefTy) {
    LDBG("  Failed: transfer_read did not read from a memref");
    return failure();
  }

  // Make sure transfer_read has no complex attributes
  if (!oldReadOp.getPermutationMapAttr().isIdentity()) {
    LDBG("  Failed: transfer_read contains complex permutation_map");
    return failure();
  }
  if (oldReadOp.isMasked()) {
    LDBG("  Failed: transfer_read is masked");
    return failure();
  }
  if (oldReadOp.hasOutOfBoundsDim()) {
    LDBG("  Failed: transfer_read has out of bounds dims");
    return failure();
  }

  // Convert DenseI64MapAttr to AffineMapAttr for memref::TransposeOp
  auto perm_I64Array = op.getPermutation();
  auto perm_AffineMap =
      AffineMap::getPermutationMap(perm_I64Array, rewriter.getContext());
  auto perm_AffineMapAttr = AffineMapAttr::get(perm_AffineMap);

  // Apply transpose to indices
  auto indices_old = oldReadOp.getIndices();
  SmallVector<Value> indices_new(indices_old.size());
  for (int64_t i = 0; i < rank; i++) {
    indices_new[perm_I64Array[i]] = indices_old[i];
  }

  // Transpose memref
  rewriter.setInsertionPointAfter(oldReadOp);
  auto transposedMemRef = rewriter.create<memref::TransposeOp>(
      op.getLoc(), memRef, perm_AffineMapAttr);

  // Create new transfer_read
  auto newReadOp = rewriter.create<vector::TransferReadOp>(
      oldReadOp.getLoc(), cast<VectorType>(output.getType()), transposedMemRef,
      indices_new, oldReadOp.getPadding(), SmallVector<bool>(rank, true));
  rewriter.replaceOp(op, newReadOp);

  LDBG("  Success, new transfer_read: " << newReadOp);
  return success();
}

LogicalResult mergeTransposeWithMemAccess(vector::TransposeOp op,
                                          PatternRewriter &rewriter) {
  LDBG("Attempt to merge transpose with memory access: " << op);

  Value input = op.getOperand();
  VectorType inputTy = cast<VectorType>(input.getType());
  Value output = op.getResult();
  VectorType outputTy = cast<VectorType>(output.getType());

  /// Try to merge transpose with transfer_write
  // If transpose has multiple uses rather than a single transfer_write, it
  // cannot be safely merged into a single transfer_write. Fallback to other
  // strategies in such cases.
  if (output.hasOneUse()) {
    if (auto writeOp =
            dyn_cast<vector::TransferWriteOp>(*output.user_begin())) {
      return transposeWhenWriting(op, rewriter);
    }
  }

  /// Try to merge transpose with transfer_read
  // 'transposeWhenReading' creates a new transfer_read which will be used by
  // users of transpose. Therefore, single use check is not required.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(input.getDefiningOp());
      readOp) {
    return transposeWhenReading(op, rewriter);
  }

  /// Create a buffer so that we can do transpose through memory
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto loc = op.getLoc();
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
    return transposeWhenWriting(op, rewriter);
  }
}

// Lower vector.transpose if a high-performance transpose implementation for the
// target backend exists.
LogicalResult convertTranspose(vector::TransposeOp op,
                               PatternRewriter &rewriter) {
  auto arch = getCpuArch();
  auto cpuFeatures = getCpuFeatures();

  if (arch == "riscv64") {
    if (cpuFeatures.find("v") != cpuFeatures.end()) {
      // RVV can use vector strided segmented load/store to process transposed
      // data.
      if (auto convertResult = mergeTransposeWithMemAccess(op, rewriter);
          succeeded(convertResult)) {
        return convertResult;
      }
    }
  }

  return failure();
}

struct ConvertTransposeOp
    : public triton::cpu::impl::ConvertTransposeOpBase<ConvertTransposeOp> {
  ConvertTransposeOp() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<vector::TransposeOp> candidates;
    mod->walk([this, &candidates](vector::TransposeOp op) {
      candidates.push_back(op);
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate);
      if (succeeded(convertTranspose(candidate, rewriter))) {
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertTransposeOp() {
  return std::make_unique<ConvertTransposeOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
