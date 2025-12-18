#include "triton/Tools/Sys/GetEnv.hpp"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::triton::cpu {
namespace rvv {

// Get RVV VSCALE.
//
// VSCALE is a implement-defined value. Program can only know that value at
// runtime. In triton, we use JIT, so we may be possible to get VSCALE when
// compiling. We support three ways to specific VSCALE using environment
// variable RVV_VLEN:
//
// 1) When RVV_VLEN is an integer, compiler uses RVV_VLEN/64 as VSCALE. (If it
// is a valid VLEN: RVV_VLEN is power of 2 && RVV_VLEN >= 64)
// 2) When RVV_VLEN is "dynamic", compiler uses a runtime call @vector.vscale to
// get VSCALE.
// 3) When RVV_VLEN is "local", compiler executes assembly `csrr %0, vlenb` to
// get VSCALE.
//
// If RVV_VLEN is undefined or invalid(including RVV_VLEN is "local" but the
// compiler is running on a machine that doesn't support RISCV-V-Extension),
// compiler will fallback and tries the following ways one by one:
// - Executes assembly `csrr %0, vlenb` to get VSCALE if possible.
// - Return a runtime call @vector.vscale.

static inline int64_t tryReadVlenb() {
#ifdef __riscv_vector
  int64_t vlenb;
  asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  return vlenb;
#else
  return -1;
#endif
}

static inline int64_t getVlen() {
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

static inline int64_t getMinimumVlen() {
  int64_t vlen = getVlen();
  if (vlen < 0)
    vlen = 64;
  return vlen;
}

// Returns: VSCALE of type 'index'
static inline Value getVscale(Location loc, PatternRewriter &rewriter) {
  int vlen = getVlen();
  if (vlen > 0) {
    return rewriter.create<arith::ConstantIndexOp>(loc, vlen / 64);
  } else {
    return rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
  }
}

static inline int64_t getBaseVlmax(Type elemTy) {
  int elemBits = elemTy.getIntOrFloatBitWidth();
  return 64 / elemBits;
}

static inline Value getVlmax(Location loc, PatternRewriter &rewriter,
                             Type elemTy) {
  Value vscale = getVscale(loc, rewriter);
  int64_t baseVlmax = getBaseVlmax(elemTy);
  Value baseVlmax_cIndex =
      rewriter.create<arith::ConstantIndexOp>(loc, baseVlmax);
  Value vlmax =
      rewriter.createOrFold<arith::MulIOp>(loc, vscale, baseVlmax_cIndex);
  return vlmax;
}

FailureOr<VectorType> getSmallestScalableTypeThatHolds(VectorType vecTy);

namespace intrinsic {

Type getRvvTupleType(PatternRewriter &rewriter, Type vecTy, unsigned int size);
Value insertToRvvTuple(Location loc, PatternRewriter &rewriter, Value tuple,
                       Value vec, int64_t index);
Value extractFromRvvTuple(Location loc, PatternRewriter &rewriter, Value tuple,
                          int64_t index);

Value createLoad(Location loc, PatternRewriter &rewriter, VectorType resTy,
                 Value basePtr, Value vl);
void createStoreMasked(Location loc, PatternRewriter &rewriter, Value basePtr,
                       Value val, Value mask, Value vl);
Value createRgather(Location loc, PatternRewriter &rewriter, Value table,
                    Value indices, Value vl);
Value createLoadStridedSegment(Location loc, PatternRewriter &rewriter,
                               int64_t numFields, VectorType vecTy, Value base,
                               Value stride, Value vl);
void createStoreStridedSegment(Location loc, PatternRewriter &rewriter,
                               Value valueToStore, Value base, Value stride,
                               Value vl);

} // namespace intrinsic

} // namespace rvv

std::string getCpuArch();
std::set<std::string> getCpuFeatures();

Value createBroadcast(Location loc, PatternRewriter &rewriter,
                      VectorType vectorTy, Value scalar);

Value createExtuiOrTrunc(Location loc, PatternRewriter &rewriter, Type targetTy,
                         Value val);

Value createMemRefToRawPtr(Location loc, PatternRewriter &rewriter,
                           Value memref);

Value convertToScalableVector(Location loc, PatternRewriter &rewriter,
                              Value fixedVector, VectorType scalableVecTy);
Value convertToFixedVector(Location loc, PatternRewriter &rewriter,
                           Value scalableVector, VectorType fixedVecTy);

Value createI64(Location loc, PatternRewriter &rewriter, int64_t val);

Value createGep(Location loc, PatternRewriter &rewriter, Value ptr, Type elemTy,
                Value index);

} // namespace mlir::triton::cpu