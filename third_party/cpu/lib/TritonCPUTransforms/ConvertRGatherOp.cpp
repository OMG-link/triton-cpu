#include "Utils.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-cpu-transforms-convert-rgather-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTRGATHEROP
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

LogicalResult convertToVectorGather(RGatherOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();

  Value table = op.getSrc();
  Value indexVec = op.getIndices();

  VectorType tableTy = cast<VectorType>(table.getType());
  VectorType resultTy = cast<VectorType>(op.getResult().getType());

  // Move 'table' to stack so 'vector.gather' can gather data from memory.
  MemRefType tableMemRefTy =
      MemRefType::get(tableTy.getShape(), tableTy.getElementType());
  Value tableMemRef = rewriter.create<memref::AllocaOp>(loc, tableMemRefTy);
  auto transferWriteIndices = SmallVector<Value>(
      tableTy.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
  rewriter.create<vector::TransferWriteOp>(loc, table, tableMemRef,
                                           transferWriteIndices);

  /// Create other arguments required by 'vector.gather'
  // Indices to index the starting point of gather. The starting point is always
  // the very beginning of table register.
  SmallVector<Value> indices(tableTy.getRank(),
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
  // No mask is needed. Set every bit to 1.
  VectorType maskType =
      VectorType::get(resultTy.getShape(), rewriter.getI1Type());
  DenseElementsAttr maskAttr =
      DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true));
  Value mask =
      rewriter.create<mlir::arith::ConstantOp>(loc, maskType, maskAttr);
  // No mask is needed. Value of 'passThru' does not matters.
  Value passThru = rewriter.create<ub::PoisonOp>(loc, resultTy);

  rewriter.replaceOpWithNewOp<vector::GatherOp>(
      op, resultTy, tableMemRef, indices, indexVec, mask, passThru);
  return success();
}

static inline int64_t nextPowerOf2(int64_t x) {
  if (x <= 1)
    return 1;
  return 1ll << (64 - __builtin_clzll(x - 1));
}

static inline Value extendVector(Location loc, PatternRewriter &rewriter,
                                 Value src, int64_t dstSize) {
  VectorType srcTy = cast<VectorType>(src.getType());
  assert(srcTy.getRank() == 1 && srcTy.isScalable() == false);
  int64_t srcSize = srcTy.getDimSize(0);
  assert(srcSize <= dstSize);
  if (srcSize == dstSize)
    return src;
  VectorType dstTy =
      VectorType::get({dstSize}, srcTy.getElementType(), {false});
  Value background = rewriter.create<ub::PoisonOp>(loc, dstTy);
  return rewriter.create<vector::InsertStridedSliceOp>(
      loc, src, background, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
}

static inline Value truncVector(Location loc, PatternRewriter &rewriter,
                                Value src, int64_t dstSize) {
  VectorType srcTy = cast<VectorType>(src.getType());
  assert(srcTy.getRank() == 1 && srcTy.isScalable() == false);
  int64_t srcSize = srcTy.getDimSize(0);
  assert(srcSize >= dstSize);
  if (srcSize == dstSize)
    return src;
  VectorType dstTy =
      VectorType::get({dstSize}, srcTy.getElementType(), {false});
  return rewriter.create<vector::ExtractStridedSliceOp>(
      loc, src, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{dstSize},
      ArrayRef<int64_t>{1});
}

LogicalResult convertToRvvIntrinsic(RGatherOp op, PatternRewriter &rewriter) {
  LDBG("Attempt to lower with RVV intrinsic: " << op);

  Location loc = op.getLoc();
  Value table = op.getSrc();
  Value indices = op.getIndices();

  VectorType tableTy = cast<VectorType>(table.getType());
  VectorType indicesTy = cast<VectorType>(indices.getType());
  VectorType resultTy = cast<VectorType>(op.getResult().getType());

  int64_t indicesSize = indicesTy.getDimSize(0);
  int64_t tableSize = tableTy.getDimSize(0);
  // LLVM IR intrinsic requires indicesSize and tableSize to be the same
  int64_t intrinsicSize = nextPowerOf2(std::max(indicesSize, tableSize));

  /// Pre-check
  // By definition of vrgather, tableTy must be 1-D vector.
  // We only need to check indices type.
  if (indicesTy.getRank() != 1) {
    LDBG("  RVV intrinsic lowering failed: indices must be 1-D vector.");
    return failure();
  }
  // Triton frontend will not produce scalable inputs.
  // Bail out when seeing scalable vectors for simplicity.
  if (tableTy.isScalable() || indicesTy.isScalable() || resultTy.isScalable()) {
    LDBG(
        "  RVV intrinsic lowering failed: scalable vectors are not supported.");
    return failure();
  }
  // If the indices or result cannot be held with 8 VREGs, bail out.
  int64_t vlen = rvv::getVlen();
  if (vlen < 0) {
    vlen = 64;
  }
  if (intrinsicSize * indicesTy.getElementTypeBitWidth() > vlen * 8) {
    LDBG("  RVV intrinsic lowering failed: indices needs more than 8 VREGs.");
    return failure();
  }
  if (intrinsicSize * resultTy.getElementTypeBitWidth() > vlen * 8) {
    LDBG("  RVV intrinsic lowering failed: result needs more than 8 VREGs.");
    return failure();
  }

  /// Prepare arguments for intrinsic
  // Extend vectors to intrinsic size
  table = extendVector(loc, rewriter, table, intrinsicSize);
  indices = extendVector(loc, rewriter, indices, intrinsicSize);
  // 'vl' limits the number of indices to be queried
  Value vl = rewriter.create<arith::ConstantIntOp>(loc, indicesSize,
                                                   rewriter.getI64Type());

  /// Create intrinsic
  VectorType intrinsicResultTy =
      VectorType::get({intrinsicSize}, resultTy.getElementType(), {false});
  Value rgatherIntrinsic = rvv::intrinsic::createRgather(
      loc, rewriter, intrinsicResultTy, table, indices, vl);
  Value result = truncVector(loc, rewriter, rgatherIntrinsic, indicesSize);
  rewriter.replaceOp(op, result);

  LDBG("  RVV intrinsic lowering succeed.");
  return success();
}

// Select a lowering strategy for RGather.
// Prefer backend-specific intrinsics if available; otherwise use the generic
// 'vector.gather' lowering.
LogicalResult convertRGather(RGatherOp op, PatternRewriter &rewriter) {
  auto arch = getCpuArch();
  auto cpuFeatures = getCpuFeatures();

  if (arch == "riscv64") {
    if (cpuFeatures.find("v") != cpuFeatures.end()) {
      if (auto convertResult = convertToRvvIntrinsic(op, rewriter);
          succeeded(convertResult)) {
        return convertResult;
      }
    }
  }

  return convertToVectorGather(op, rewriter);
}

struct ConvertRGatherOp
    : public triton::cpu::impl::ConvertRGatherOpBase<ConvertRGatherOp> {
  ConvertRGatherOp() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<RGatherOp> candidates;
    mod->walk([this, &candidates](RGatherOp op) {
      candidates.push_back(op);
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate);
      if (succeeded(convertRGather(candidate, rewriter))) {
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertRGatherOp() {
  return std::make_unique<ConvertRGatherOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
