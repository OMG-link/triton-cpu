#include "Utils.h"
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
  Location loc = op.getLoc();
  Value table = op.getSrc();
  Value indices = op.getIndices();

  VectorType tableTy = cast<VectorType>(table.getType());
  VectorType indicesTy = cast<VectorType>(indices.getType());
  VectorType resultTy = cast<VectorType>(op.getResult().getType());

  /// Pre-check
  // By definition of vrgather, tableTy must be 1-D vector.
  // We only need to check indices type.
  if (indicesTy.getRank() != 1) {
    LDBG("  RVV intrinsic lowering failed: indices must be 1-D vector.");
    return failure();
  }

  /// Store vectors to memory
  // We need to call RVV intrinsic later, which requires scalable vector.
  // To cast fixed vector to scalable vector, we need transition via memory.
  Value tablePtr;
  {
    MemRefType tableMemRefTy =
        MemRefType::get(tableTy.getShape(), tableTy.getElementType());
    Value tableMemRef = rewriter.create<memref::AllocaOp>(loc, tableMemRefTy);
    auto transferWriteIndices = SmallVector<Value>(
        tableTy.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
    rewriter.create<vector::TransferWriteOp>(loc, table, tableMemRef,
                                             transferWriteIndices);
    Value tablePtr_index =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                tableMemRef);
    Value tablePtr_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), tablePtr_index);
    tablePtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), tablePtr_i64);
  }
  Value indicesPtr;
  {
    MemRefType indicesMemRefTy =
        MemRefType::get(indicesTy.getShape(), indicesTy.getElementType());
    Value indicesMemRef =
        rewriter.create<memref::AllocaOp>(loc, indicesMemRefTy);
    auto transferWriteIndices = SmallVector<Value>(
        indicesTy.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
    rewriter.create<vector::TransferWriteOp>(loc, indices, indicesMemRef,
                                             transferWriteIndices);
    Value indicesPtr_index =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                indicesMemRef);
    Value indicesPtr_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), indicesPtr_index);
    indicesPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), indicesPtr_i64);
  }
  Value resultPtr, resultMemRef;
  {
    MemRefType resultMemRefTy =
        MemRefType::get(resultTy.getShape(), resultTy.getElementType());
    resultMemRef = rewriter.create<memref::AllocaOp>(loc, resultMemRefTy);
    Value resultPtr_index =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                resultMemRef);
    Value resultPtr_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), resultPtr_index);
    resultPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), resultPtr_i64);
  }

  /// Create loops to iterate over table chunk and indices chunk.
  // Currently, we only supprt M1 vrgather.
  int64_t baseVlmax = rvv::getBaseVlmax(tableTy.getElementType());
  Value vlmax = rewriter.createOrFold<arith::MulIOp>(
      loc, rewriter.create<arith::ConstantIndexOp>(loc, baseVlmax),
      rvv::getVscale(loc, rewriter));
  Value vlmax_i64 =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), vlmax);
  Value tableNumElements =
      rewriter.create<arith::ConstantIndexOp>(loc, tableTy.getNumElements());
  if (tableTy.isScalable()) {
    tableNumElements = rewriter.createOrFold<arith::MulIOp>(
        loc, tableNumElements, rvv::getVscale(loc, rewriter));
  }
  Value indicesNumElements =
      rewriter.create<arith::ConstantIndexOp>(loc, indicesTy.getNumElements());
  if (indicesTy.isScalable()) {
    indicesNumElements = rewriter.createOrFold<arith::MulIOp>(
        loc, indicesNumElements, rvv::getVscale(loc, rewriter));
  }
  VectorType subTableTy =
      VectorType::get({baseVlmax}, tableTy.getElementType(), {true});
  VectorType subIndicesTy =
      VectorType::get({baseVlmax}, indicesTy.getElementType(), {true});
  VectorType subResultTy =
      VectorType::get({baseVlmax}, resultTy.getElementType(), {true});
  auto forIndicesOp = rewriter.create<scf::ForOp>(
      loc, rewriter.create<arith::ConstantIndexOp>(loc, 0), indicesNumElements,
      vlmax);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forIndicesOp.getBody());
    Value indicesOffset = forIndicesOp.getInductionVar();
    Value indicesOffset_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), indicesOffset);
    Value indicesLeft = rewriter.createOrFold<arith::SubIOp>(
        loc, indicesNumElements, indicesOffset);
    Value indicesVl =
        rewriter.createOrFold<arith::MinUIOp>(loc, vlmax, indicesLeft);
    Value indicesVl_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), indicesVl);

    Value subIndicesPtr = rewriter.createOrFold<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
        subIndicesTy.getElementType(), indicesPtr, indicesOffset_i64);
    Value subIndices = rvv::intrinsic::createLoad(loc, rewriter, subIndicesTy,
                                                  subIndicesPtr, indicesVl_i64);
    Value vlmax_v = createBroadcast(
        loc, rewriter, subIndicesTy,
        createExtuiOrTrunc(loc, rewriter, subIndicesTy.getElementType(),
                           vlmax_i64));
    Value subIndicesGroupId =
        rewriter.createOrFold<arith::DivUIOp>(loc, subIndices, vlmax_v);
    Value subIndicesRegId =
        rewriter.createOrFold<arith::RemUIOp>(loc, subIndices, vlmax_v);

    auto forTableOp = rewriter.create<scf::ForOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 0), tableNumElements,
        vlmax);
    {
      OpBuilder::InsertionGuard guard2(rewriter);
      rewriter.setInsertionPointToStart(forTableOp.getBody());
      Value tableOffset = forTableOp.getInductionVar();
      Value tableOffset_i64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), tableOffset);
      Value tableLeft = rewriter.createOrFold<arith::SubIOp>(
          loc, tableNumElements, tableOffset);
      Value tableVl =
          rewriter.createOrFold<arith::MinUIOp>(loc, vlmax, tableLeft);
      Value tableVl_i64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), tableVl);

      Value tableGroupId =
          rewriter.createOrFold<arith::DivUIOp>(loc, tableOffset, vlmax);
      Value tableGroupId_i64 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), tableGroupId);
      Value mask = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, subIndicesGroupId,
          createBroadcast(loc, rewriter, subIndicesTy,
                          createExtuiOrTrunc(loc, rewriter,
                                             subIndicesTy.getElementType(),
                                             tableGroupId_i64)));

      Value subTablePtr = rewriter.createOrFold<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          subTableTy.getElementType(), tablePtr, tableOffset_i64);
      Value subTable = rvv::intrinsic::createLoad(loc, rewriter, subTableTy,
                                                  subTablePtr, tableVl_i64);

      Value subResult = rvv::intrinsic::createRgather(
          loc, rewriter, subResultTy, subTable, subIndicesRegId, indicesVl_i64);
      Value subResultPtr = rewriter.createOrFold<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          subResultTy.getElementType(), resultPtr, indicesOffset_i64);
      rvv::intrinsic::createStoreMasked(loc, rewriter, subResult, subResultPtr,
                                        mask, indicesVl_i64);
    }
  }

  /// Load result from memory
  Value finalResult = rewriter.create<vector::TransferReadOp>(
      loc, resultTy, resultMemRef,
      SmallVector<Value>(resultTy.getRank(),
                         rewriter.create<arith::ConstantIndexOp>(loc, 0)));
  rewriter.replaceOp(op, finalResult);

  return success();
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
