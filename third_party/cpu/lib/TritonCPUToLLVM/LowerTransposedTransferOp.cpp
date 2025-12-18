#include "../TritonCPUTransforms/Utils.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-cpu-to-llvm-lower-transposed-transfer-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_LOWERTRANSPOSEDTRANSFEROP
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct TransposedTransferCandidate {
  Operation *op;
};

template <typename TransferOp> bool isTransposedTransfer(Operation *op) {
  TransferOp transferOp = dyn_cast<TransferOp>(op);
  if (!transferOp)
    return false;
  LDBG("Checking candidate: " << *op);
  bool isOutOfBounds = transferOp.hasOutOfBoundsDim();
  bool isMasked = bool(transferOp.getMask());
  bool isPermuted = !transferOp.getPermutationMap().isIdentity();
  if (isOutOfBounds || isMasked || isPermuted) {
    // In fact, we can handle permuted cases.
    // But skip it for simplicity for now.
    LDBG("  Skipped: 'isOutOfBounds || isMasked || isPermuted'");
    return false;
  }
  auto memRefTy = dyn_cast<MemRefType>(transferOp.getSource().getType());
  if (!memRefTy) {
    LDBG("  Skipped: '!memRefTy'");
    return false;
  }
  auto strides = memRefTy.getStridesAndOffset().first;
  int64_t rank = strides.size();
  if (rank != 2) {
    LDBG("  Skipped: 'rank != 2'");
    return false;
  }
  LDBG("  Final check: " << strides[rank - 2] << " " << strides[rank - 1]);
  return strides[rank - 2] == 1 && strides[rank - 1] != 1;
}

static Value getLastDimStride(Value memRef, Location loc,
                              PatternRewriter &rewriter) {
  auto memRefTy = cast<MemRefType>(memRef.getType());
  auto memRefMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, memRef);
  return memRefMetadata.getStrides().back();
}

LogicalResult convertTransposedReadToRvv(vector::TransferReadOp readOp,
                                         PatternRewriter &rewriter) {
  LDBG("  Lowering to RVV intrinsic");
  Location loc = readOp.getLoc();
  VectorType matTy = readOp.getResult().getType();
  int64_t m = matTy.getDimSize(0);
  int64_t k = matTy.getDimSize(1);
  Value memRef = readOp.getSource();
  Value stride_index = getLastDimStride(memRef, loc, rewriter);
  int64_t vlen = rvv::getMinimumVlen();

  if (matTy.isScalable()) {
    LDBG("  Failed: scalable dim are not allowed");
    return failure();
  }
  if (std::min(m, int64_t(4)) * k * matTy.getElementTypeBitWidth() > 8 * vlen) {
    // FIXME: we can work around by spliting vectors
    LDBG("  Failed: row too long");
    return failure();
  }

  VectorType rowType_fixed = VectorType::get({k}, matTy.getElementType());
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
  Value rawPtr = createMemRefToRawPtr(loc, rewriter, memRef);
  Value stride_i64 = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), stride_index);

  for (int64_t i_m = 0; i_m < m; i_m += 4) {
    int64_t mr = std::min(m - i_m, int64_t(4));
    Value ptr = createGep(loc, rewriter, rawPtr, matTy.getElementType(),
                          createI64(loc, rewriter, i_m));
    Value mrTuple = rvv::intrinsic::createLoadStridedSegment(
        loc, rewriter, mr, rowType_scalable, ptr, stride_i64, vl);
    for (int64_t i_mr = 0; i_mr < mr; i_mr++) {
      Value row_scalable =
          rvv::intrinsic::extractFromRvvTuple(loc, rewriter, mrTuple, i_mr);
      Value row_fixed =
          convertToFixedVector(loc, rewriter, row_scalable, rowType_fixed);
      result =
          rewriter.create<vector::InsertOp>(loc, row_fixed, result, i_m + i_mr);
    }
  }

  rewriter.replaceOp(readOp, result);
  LDBG("  Success.");
  return success();
}

LogicalResult convertTransposedWriteToRvv(vector::TransferWriteOp writeOp,
                                          PatternRewriter &rewriter) {
  LDBG("  Lowering to RVV intrinsic");
  Location loc = writeOp.getLoc();
  Value mat = writeOp.getVector();
  VectorType matTy = writeOp.getVector().getType();
  Type matElemTy = matTy.getElementType();
  int64_t m = matTy.getDimSize(0);
  int64_t k = matTy.getDimSize(1);
  Value memRef = writeOp.getSource();
  Value stride_index = getLastDimStride(memRef, loc, rewriter);
  int64_t vlen = rvv::getMinimumVlen();

  if (matTy.isScalable()) {
    LDBG("  Failed: scalable dim are not allowed");
    return failure();
  }
  if (std::min(m, int64_t(4)) * k * matTy.getElementTypeBitWidth() > 8 * vlen) {
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
  Value stride_i64 = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), stride_index);

  // Calculate the starting point of matrix
  auto indices = writeOp.getIndices();
  Value basePtr = createMemRefToRawPtr(loc, rewriter, memRef);
  Value offset = rewriter.create<arith::AddIOp>(
      loc, indices[0],
      rewriter.create<arith::MulIOp>(loc, indices[1], stride_index));
  offset =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), offset);
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

  rewriter.eraseOp(writeOp);
  LDBG("  Success.");
  return success();
}

LogicalResult convertToRvv(TransposedTransferCandidate candidate,
                           PatternRewriter &rewriter) {
  if (auto readOp = dyn_cast<vector::TransferReadOp>(candidate.op)) {
    return convertTransposedReadToRvv(readOp, rewriter);
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(candidate.op)) {
    return convertTransposedWriteToRvv(writeOp, rewriter);
  }
  llvm_unreachable("This is not a candidate.");
}

LogicalResult lowerTransposedTransfer(TransposedTransferCandidate candidate,
                                      PatternRewriter &rewriter) {
  auto arch = getCpuArch();
  auto cpuFeatures = getCpuFeatures();

  if (arch == "riscv64") {
    if (cpuFeatures.find("v") != cpuFeatures.end()) {
      if (auto convertResult = convertToRvv(candidate, rewriter);
          succeeded(convertResult)) {
        return convertResult;
      }
    }
  }

  return failure();
}

struct LowerTransposedTransferOp
    : public triton::cpu::impl::LowerTransposedTransferOpBase<
          LowerTransposedTransferOp> {
  LowerTransposedTransferOp() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<TransposedTransferCandidate> candidates;
    mod->walk([this, &candidates](Operation *op) {
      if (isTransposedTransfer<vector::TransferReadOp>(op) ||
          isTransposedTransfer<vector::TransferWriteOp>(op)) {
        candidates.push_back({op});
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << *candidate.op);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(lowerTransposedTransfer(candidate, rewriter))) {
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

std::unique_ptr<OperationPass<ModuleOp>> createLowerTransposedTransferOp() {
  return std::make_unique<LowerTransposedTransferOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
