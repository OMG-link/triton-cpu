#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/TargetParser/Host.h"

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

LogicalResult convertToRvvIntrinsic(RGatherOp op, PatternRewriter &rewriter) {
  // TODO
  return failure();
}

// Select a lowering strategy for RGather.
// Prefer backend-specific intrinsics if available; otherwise use the generic
// 'vector.gather' lowering.
LogicalResult convertRGather(RGatherOp op, PatternRewriter &rewriter) {
  auto getCpuArch = []() -> std::string {
    std::string triple = llvm::sys::getProcessTriple();
    std::size_t pos = triple.find('-');
    if (pos == std::string::npos) {
      return "unknown";
    }
    std::string arch = triple.substr(0, pos);
    return arch;
  };
  auto getCpuFeatures = []() -> std::set<std::string> {
    auto features = llvm::sys::getHostCPUFeatures();
    std::set<std::string> res;
    for (auto &f : features) {
      if (f.second)
        res.insert(f.first().str());
    }
    return res;
  };

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
