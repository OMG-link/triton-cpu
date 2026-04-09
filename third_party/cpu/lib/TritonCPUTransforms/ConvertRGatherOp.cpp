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
  Value tableMemRef = memref::AllocaOp::create(rewriter, loc, tableMemRefTy);
  auto transferWriteIndices = SmallVector<Value>(
      tableTy.getRank(), arith::ConstantIndexOp::create(rewriter, loc, 0));
  vector::TransferWriteOp::create(rewriter, loc, table, tableMemRef,
                                  transferWriteIndices);

  /// Create other arguments required by 'vector.gather'
  // Indices to index the starting point of gather. The starting point is always
  // the very beginning of table register.
  SmallVector<Value> indices(tableTy.getRank(),
                             arith::ConstantIndexOp::create(rewriter, loc, 0));
  // No mask is needed. Set every bit to 1.
  VectorType maskType =
      VectorType::get(resultTy.getShape(), rewriter.getI1Type());
  DenseElementsAttr maskAttr =
      DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true));
  Value mask =
      mlir::arith::ConstantOp::create(rewriter, loc, maskType, maskAttr);
  // No mask is needed. Value of 'passThru' does not matters.
  Value passThru = ub::PoisonOp::create(rewriter, loc, resultTy);

  rewriter.replaceOpWithNewOp<vector::GatherOp>(
      op, resultTy, tableMemRef, indices, indexVec, mask, passThru);
  return success();
}

static inline int64_t nextPowerOf2(int64_t x) {
  if (x <= 1)
    return 1;
  return 1ll << (64 - __builtin_clzll(x - 1));
}

LogicalResult convertToRvvIntrinsic(RGatherOp op, PatternRewriter &rewriter) {
  LDBG("Attempt to lower with RVV intrinsic: " << op);

  Location loc = op.getLoc();
  Value table_fixed = op.getSrc();
  Value indices_fixed = op.getIndices();

  VectorType tableTy_fixed = cast<VectorType>(table_fixed.getType());
  VectorType indicesTy_fixed = cast<VectorType>(indices_fixed.getType());
  VectorType resultTy_fixed = cast<VectorType>(op.getResult().getType());

  // LLVM IR intrinsic requires indicesSize and tableSize to be the same
  int64_t intrinsicSize = nextPowerOf2(
      std::max(indicesTy_fixed.getDimSize(0), tableTy_fixed.getDimSize(0)));

  /// Pre-check
  // By definition of vrgather, tableTy must be 1-D vector.
  // We only need to check indices type.
  if (indicesTy_fixed.getRank() != 1) {
    LDBG("  RVV intrinsic lowering failed: indices must be 1-D vector.");
    return failure();
  }
  // Triton frontend will not produce scalable inputs.
  // Bail out when seeing scalable vectors for simplicity.
  if (tableTy_fixed.isScalable() || indicesTy_fixed.isScalable() ||
      resultTy_fixed.isScalable()) {
    LDBG(
        "  RVV intrinsic lowering failed: scalable vectors are not supported.");
    return failure();
  }
  // If the indices or result cannot be held with 8 VREGs, bail out.
  int64_t vlen = rvv::getMinimumVlen();
  if (intrinsicSize * indicesTy_fixed.getElementTypeBitWidth() > vlen * 8) {
    LDBG("  RVV intrinsic lowering failed: indices needs more than 8 VREGs.");
    return failure();
  }
  if (intrinsicSize * resultTy_fixed.getElementTypeBitWidth() > vlen * 8) {
    LDBG("  RVV intrinsic lowering failed: result needs more than 8 VREGs.");
    return failure();
  }

  /// Prepare arguments for intrinsic
  // Extend vectors to intrinsic size
  VectorType resultTy_scalable =
      rvv::getSmallestScalableTypeThatHolds(
          VectorType::get({intrinsicSize}, resultTy_fixed.getElementType()))
          .value();
  VectorType tableTy_scalable =
      resultTy_scalable.cloneWith(std::nullopt, tableTy_fixed.getElementType());
  VectorType indicesTy_scalable = resultTy_scalable.cloneWith(
      std::nullopt, indicesTy_fixed.getElementType());
  Value table_scalable =
      convertToScalableVector(rewriter, loc, table_fixed, tableTy_scalable);
  Value indices_scalable =
      convertToScalableVector(rewriter, loc, indices_fixed, indicesTy_scalable);

  // 'vl' limits the number of indices to be queried
  Value vl = createI64(rewriter, loc, indicesTy_fixed.getDimSize(0));

  /// Create intrinsic
  Value result_scalable = rvv::intrinsic::createRgather(
      rewriter, loc, table_scalable, indices_scalable, vl);
  Value result_fixed =
      convertToFixedVector(rewriter, loc, result_scalable, resultTy_fixed);
  rewriter.replaceOp(op, result_fixed);

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
