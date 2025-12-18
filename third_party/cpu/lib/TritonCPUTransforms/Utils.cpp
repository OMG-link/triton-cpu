#include "Utils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/TargetParser/Host.h"

namespace mlir::triton::cpu {

static inline bool is1DScalableVectorType(VectorType type) {
  return type.getRank() == 1 && type.isScalable();
}

static inline bool is1DScalableVectorType(Type type) {
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    return is1DScalableVectorType(vecTy);
  } else {
    return false;
  }
}

static inline bool is1DFixedVectorType(VectorType type) {
  return type.getRank() == 1 && !type.isScalable();
}

static inline bool is1DFixedVectorType(Type type) {
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    return is1DFixedVectorType(vecTy);
  } else {
    return false;
  }
}

static inline int64_t nextPowerOf2(int64_t x) {
  if (x <= 1)
    return 1;
  return 1ll << (64 - __builtin_clzll(x - 1));
}

namespace rvv {

int64_t getVlen() {
  std::string RVV_VLEN = mlir::triton::tools::getStrEnv("RVV_VLEN");
  // if RVV_VLEN is defined
  if (!RVV_VLEN.empty()) {
    if (RVV_VLEN == "dynamic") {
      return -1;
    } else if (RVV_VLEN == "local") {
      int vlenb = tryReadVlenb();
      if (vlenb > 0) {
        return vlenb * 8;
      } else {
        return -1;
      }
    } else {
      char *end;
      long vlen = strtol(RVV_VLEN.c_str(), &end, 10);
      if (*end == '\0') {
        if (vlen >= 64 && (vlen & (vlen - 1)) == 0) {
          return vlen;
        }
      }
    }
  }
  // fallback
  int vlenb = tryReadVlenb();
  if (vlenb > 0) {
    return vlenb * 8;
  }
  return -1;
}

FailureOr<VectorType> getSmallestScalableTypeThatHolds(VectorType vecTy) {
  assert(is1DFixedVectorType(vecTy));
  Type elemTy = vecTy.getElementType();
  int64_t elemNum = vecTy.getNumElements();
  int64_t elemBitWidth = vecTy.getElementTypeBitWidth();
  int64_t vlen = getMinimumVlen();
  if (elemNum * elemBitWidth > 8 * vlen) {
    // Cannot find suitable scalable type
    return failure();
  }
  int64_t regBits = nextPowerOf2(elemNum * elemBitWidth);
  int64_t baseRegBits = regBits * 64 / vlen;
  int64_t baseRegElemNum = baseRegBits / elemBitWidth;
  return VectorType::get({baseRegElemNum}, elemTy, {true});
}

namespace intrinsic {

Type getRvvTupleType(PatternRewriter &rewriter, Type vecTy, unsigned int size) {
  assert(is1DScalableVectorType(vecTy));
  if (size == 1) {
    return vecTy;
  } else {
    assert(2 <= size && size <= 8);
    return rewriter.getType<LLVM::LLVMTargetExtType>(
        "riscv.vector.tuple", ArrayRef{vecTy}, ArrayRef{size});
  }
}

Value insertToRvvTuple(Location loc, PatternRewriter &rewriter, Value tuple,
                       Value vec, int64_t index) {
  if (auto tupleTy = dyn_cast<LLVM::LLVMTargetExtType>(tuple.getType())) {
    VectorType vecTy = cast<VectorType>(tupleTy.getTypeParams()[0]);
    assert(is1DScalableVectorType(vec.getType()));
    assert(is1DScalableVectorType(vecTy));
    assert(vec.getType() == vecTy);
    StringAttr intrinsicName =
        rewriter.getStringAttr("llvm.riscv.tuple.insert");
    Value index_value = rewriter.create<arith::ConstantIntOp>(
        loc, index, rewriter.getI32Type());
    SmallVector<Value> args = {tuple, vec, index_value};
    auto rvvIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, tuple.getType(), intrinsicName, args);
    return rvvIntrinsicOp.getResult(0);
  } else if (auto vecTy = cast<VectorType>(tuple.getType())) {
    // When num_fields = 1, it is a vector type
    assert(index == 0);
    return vec;
  }
  llvm_unreachable("invalid tuple type");
}

Value extractFromRvvTuple(Location loc, PatternRewriter &rewriter, Value tuple,
                          int64_t index) {
  if (auto tupleTy = dyn_cast<LLVM::LLVMTargetExtType>(tuple.getType())) {
    VectorType vecTy = cast<VectorType>(tupleTy.getTypeParams()[0]);
    assert(is1DScalableVectorType(vecTy));
    StringAttr intrinsicName =
        rewriter.getStringAttr("llvm.riscv.tuple.extract");
    Value index_value = rewriter.create<arith::ConstantIntOp>(
        loc, index, rewriter.getI32Type());
    SmallVector<Value> args = {tuple, index_value};
    auto rvvIntrinsicOp =
        rewriter.create<LLVM::CallIntrinsicOp>(loc, vecTy, intrinsicName, args);
    return rvvIntrinsicOp.getResult(0);
  } else if (auto vecTy = cast<VectorType>(tuple.getType())) {
    // When num_fields = 1, it is a vector type
    assert(index == 0);
    return tuple;
  }
  llvm_unreachable("invalid tuple type");
}

Value createLoad(Location loc, PatternRewriter &rewriter, VectorType resTy,
                 Value basePtr, Value vl) {
  assert(is1DScalableVectorType(resTy));
  StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vle");
  Value poison = rewriter.create<LLVM::PoisonOp>(loc, resTy);
  SmallVector<Value> args = {poison, basePtr, vl};
  auto rvvLoadOp =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, resTy, intrinsicName, args);
  return rvvLoadOp.getResult(0);
}

void createStoreMasked(Location loc, PatternRewriter &rewriter, Value val,
                       Value basePtr, Value mask, Value vl) {
  assert(is1DScalableVectorType(val.getType()));
  StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vse.mask");
  SmallVector<Value> args = {val, basePtr, mask, vl};
  auto rvvStoreOp =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, intrinsicName, args);
  return;
}

Value createRgather(Location loc, PatternRewriter &rewriter, Value table,
                    Value indices, Value vl) {
  VectorType tableTy = cast<VectorType>(table.getType());
  VectorType indicesTy = cast<VectorType>(indices.getType());
  assert(is1DScalableVectorType(tableTy));
  assert(is1DScalableVectorType(indicesTy));
  VectorType resTy =
      indicesTy.cloneWith(std::nullopt, tableTy.getElementType());
  // LLVM IR intrinsic requires all operands has the same number of elements.
  assert(tableTy.getNumElements() == indicesTy.getNumElements());
  if (tableTy.getElementTypeBitWidth() == indicesTy.getElementTypeBitWidth()) {
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

Value createLoadStridedSegment(Location loc, PatternRewriter &rewriter,
                               int64_t numFields, VectorType vecTy, Value base,
                               Value stride, Value vl) {
  assert(is1DScalableVectorType(vecTy));
  assert(numFields * vecTy.getDimSize(0) * vecTy.getElementTypeBitWidth() <=
         8 * getMinimumVlen());
  if (numFields == 1) {
    StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vlse");
    Type intrinsicRetTy = vecTy;
    Value poison = rewriter.create<ub::PoisonOp>(loc, vecTy);
    SmallVector<Value> args = {poison, base, stride, vl};
    auto rvvIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, intrinsicRetTy, intrinsicName, args);
    return rvvIntrinsicOp.getResult(0);
  } else {
    assert(2 <= numFields && numFields <= 8);
    Type groupType = getRvvTupleType(rewriter, vecTy, (unsigned int)numFields);
    Type intrinsicRetTy = groupType;
    StringAttr intrinsicName = rewriter.getStringAttr(
        std::string("llvm.riscv.vlsseg") + std::to_string(numFields));
    Value poison = rewriter.create<ub::PoisonOp>(loc, groupType);
    Value tama =
        rewriter.create<arith::ConstantIntOp>(loc, 3, rewriter.getI64Type());
    SmallVector<Value> args = {poison, base, stride, vl, tama};
    auto rvvIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, intrinsicRetTy, intrinsicName, args);
    return rvvIntrinsicOp.getResult(0);
  }
}

void createStoreStridedSegment(Location loc, PatternRewriter &rewriter,
                               Value valueToStore, Value base, Value stride,
                               Value vl) {
  if (auto vecTy = dyn_cast<VectorType>(valueToStore.getType())) {
    StringAttr intrinsicName = rewriter.getStringAttr("llvm.riscv.vsse");
    Type intrinsicRetTy = rewriter.getType<LLVM::LLVMVoidType>();
    SmallVector<Value> args = {valueToStore, base, stride, vl};
    auto rvvIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, intrinsicRetTy, intrinsicName, args);
  } else {
    auto groupType = cast<LLVM::LLVMTargetExtType>(valueToStore.getType());
    vecTy = cast<VectorType>(groupType.getTypeParams()[0]);
    int64_t numFields = groupType.getIntParams()[0];
    assert(is1DScalableVectorType(vecTy));
    assert(numFields * vecTy.getDimSize(0) * vecTy.getElementTypeBitWidth() <=
           8 * getMinimumVlen());
    StringAttr intrinsicName = rewriter.getStringAttr(
        std::string("llvm.riscv.vssseg") + std::to_string(numFields));
    Type intrinsicRetTy = rewriter.getType<LLVM::LLVMVoidType>();
    Value tama =
        rewriter.create<arith::ConstantIntOp>(loc, 3, rewriter.getI64Type());
    SmallVector<Value> args = {valueToStore, base, stride, vl, tama};
    auto rvvIntrinsicOp = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, intrinsicRetTy, intrinsicName, args);
  }
  return;
}

} // namespace intrinsic
} // namespace rvv

std::string getCpuArch() {
  std::string triple = llvm::sys::getProcessTriple();
  std::size_t pos = triple.find('-');
  if (pos == std::string::npos) {
    return "unknown";
  }
  std::string arch = triple.substr(0, pos);
  return arch;
}

std::set<std::string> getCpuFeatures() {
  auto features = llvm::sys::getHostCPUFeatures();
  std::set<std::string> res;
  for (auto &f : features) {
    if (f.second)
      res.insert(f.first().str());
  }
  return res;
}

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

Value createMemRefToRawPtr(Location loc, PatternRewriter &rewriter,
                           Value memref) {
  MLIRContext *context = rewriter.getContext();
  Value ptr_index =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value ptr_i64 = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), ptr_index);
  Value ptr = rewriter.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(context), ptr_i64);
  return ptr;
}

Value convertToScalableVector(Location loc, PatternRewriter &rewriter,
                              Value fixedVector, VectorType scalableVecTy) {
  assert(is1DFixedVectorType(fixedVector.getType()));
  assert(is1DScalableVectorType(scalableVecTy));
  Value poison = rewriter.create<ub::PoisonOp>(loc, scalableVecTy);
  return rewriter.create<LLVM::vector_insert>(loc, poison, fixedVector, 0);
}

Value convertToFixedVector(Location loc, PatternRewriter &rewriter,
                           Value scalableVector, VectorType fixedVecTy) {
  assert(is1DFixedVectorType(fixedVecTy));
  assert(is1DScalableVectorType(scalableVector.getType()));
  return rewriter.create<LLVM::vector_extract>(loc, fixedVecTy, scalableVector,
                                               0);
}

Value createI64(Location loc, PatternRewriter &rewriter, int64_t val) {
  return rewriter.create<arith::ConstantIntOp>(loc, val, rewriter.getI64Type());
}

Value createGep(Location loc, PatternRewriter &rewriter, Value ptr, Type elemTy,
                Value index) {
  return rewriter.create<LLVM::GEPOp>(loc,
                                      rewriter.getType<LLVM::LLVMPointerType>(),
                                      elemTy, ptr, ValueRange{index}, true);
}

} // namespace mlir::triton::cpu