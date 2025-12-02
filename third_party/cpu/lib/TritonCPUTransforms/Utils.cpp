#include "Utils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::cpu {

namespace rvv {
namespace intrinsic {

Value createLoad(Location loc, PatternRewriter &rewriter, VectorType resTy,
                 Value basePtr, Value vl) {
  StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vle");
  Value poison = rewriter.create<LLVM::PoisonOp>(loc, resTy);
  SmallVector<Value> args = {poison, basePtr, vl};
  auto rvvLoadOp =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, resTy, intrinsicName, args);
  return rvvLoadOp.getResult(0);
}

void createStoreMasked(Location loc, PatternRewriter &rewriter, Value val,
                       Value basePtr, Value mask, Value vl) {
  StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vse.mask");
  SmallVector<Value> args = {val, basePtr, mask, vl};
  auto rvvStoreOp =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, intrinsicName, args);
  return;
}

Value createRgather(Location loc, PatternRewriter &rewriter, VectorType resTy,
                    Value table, Value indices, Value vl) {
  VectorType tableTy = cast<VectorType>(table.getType());
  VectorType indicesTy = cast<VectorType>(indices.getType());
  if (tableTy.getElementTypeBitWidth() == indicesTy.getElementTypeBitWidth() &&
      tableTy.getNumElements() == indicesTy.getNumElements()) {
    StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vrgather.vv");
    Value poison = rewriter.create<LLVM::PoisonOp>(loc, resTy);
    SmallVector<Value> args = {poison, table, indices, vl};
    auto rvvRgatherOp =
        rewriter.create<LLVM::CallIntrinsicOp>(loc, resTy, intrinsicName, args);
    return rvvRgatherOp.getResult(0);
  } else {
    if (indicesTy.getElementTypeBitWidth() != 16) {
      indices = createExtuiOrTrunc(
          loc, rewriter,
          indicesTy.cloneWith(std::nullopt, rewriter.getI16Type()), indices);
    }
    StringAttr intrinsicName =
        rewriter.getStringAttr("llvm.riscv.vrgatherei16.vv");
    Value poison = rewriter.create<LLVM::PoisonOp>(loc, resTy);
    SmallVector<Value> args = {poison, table, indices, vl};
    auto rvvRgatherOp =
        rewriter.create<LLVM::CallIntrinsicOp>(loc, resTy, intrinsicName, args);
    return rvvRgatherOp.getResult(0);
  }
}

} // namespace intrinsic
} // namespace rvv

Value createBroadcast(Location loc, PatternRewriter &rewriter,
                      VectorType vectorTy, Value scalar) {
  assert(vectorTy.getElementType() == scalar.getType());
  return rewriter.createOrFold<vector::SplatOp>(loc, vectorTy, scalar);
}

Value createExtuiOrTrunc(Location loc, PatternRewriter &rewriter, Type targetTy,
                         Value val) {
  Type sourceType = val.getType();
  if (VectorType sourceVecTy = dyn_cast<VectorType>(sourceType),
      targetVecTy = dyn_cast<VectorType>(targetTy);
      sourceVecTy || targetVecTy) {
    assert(sourceVecTy && targetVecTy &&
           "Source and target vectors must be either all vector types or all "
           "scalar types");
    assert(sourceVecTy.getShape() == targetVecTy.getShape() &&
           "Source and target vector types must have the same shape");
    if (sourceVecTy.getElementTypeBitWidth() <
        targetVecTy.getElementTypeBitWidth()) {
      return rewriter.createOrFold<arith::ExtUIOp>(loc, targetTy, val);
    } else {
      return rewriter.createOrFold<arith::TruncIOp>(loc, targetTy, val);
    }
  } else {
    if (sourceType.getIntOrFloatBitWidth() < targetTy.getIntOrFloatBitWidth()) {
      return rewriter.createOrFold<arith::ExtUIOp>(loc, targetTy, val);
    } else {
      return rewriter.createOrFold<arith::TruncIOp>(loc, targetTy, val);
    }
  }
}

} // namespace mlir::triton::cpu