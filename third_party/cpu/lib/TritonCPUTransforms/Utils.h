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

int64_t getVlen();

static inline int64_t getMinimumVlen() {
  int64_t vlen = getVlen();
  if (vlen < 0)
    vlen = 64;
  return vlen;
}

// Returns: VSCALE of type 'index'
static inline Value getVscale(PatternRewriter &rewriter, Location loc) {
  int vlen = getVlen();
  if (vlen > 0) {
    return arith::ConstantIndexOp::create(rewriter, loc, vlen / 64);
  } else {
    return vector::VectorScaleOp::create(rewriter, loc,
                                         rewriter.getIndexType());
  }
}

static inline int64_t getBaseVlmax(Type elemTy) {
  int elemBits = elemTy.getIntOrFloatBitWidth();
  return 64 / elemBits;
}

static inline Value getVlmax(PatternRewriter &rewriter, Location loc,
                             Type elemTy) {
  Value vscale = getVscale(rewriter, loc);
  int64_t baseVlmax = getBaseVlmax(elemTy);
  Value baseVlmax_cIndex =
      arith::ConstantIndexOp::create(rewriter, loc, baseVlmax);
  Value vlmax =
      rewriter.createOrFold<arith::MulIOp>(loc, vscale, baseVlmax_cIndex);
  return vlmax;
}

FailureOr<VectorType> getSmallestScalableTypeThatHolds(VectorType vecTy);

namespace intrinsic {

Type getRvvTupleType(PatternRewriter &rewriter, Type vecTy, unsigned int size);
Value insertToRvvTuple(PatternRewriter &rewriter, Location loc, Value tuple,
                       Value vec, int64_t index);
Value extractFromRvvTuple(PatternRewriter &rewriter, Location loc, Value tuple,
                          int64_t index);

Value createLoad(PatternRewriter &rewriter, Location loc, VectorType resTy,
                 Value basePtr, Value vl);
void createStoreMasked(PatternRewriter &rewriter, Location loc, Value basePtr,
                       Value val, Value mask, Value vl);
Value createRgather(PatternRewriter &rewriter, Location loc, Value table,
                    Value indices, Value vl);
Value createLoadStridedSegment(PatternRewriter &rewriter, Location loc,
                               int64_t numFields, VectorType vecTy, Value base,
                               Value stride, Value vl);
void createStoreStridedSegment(PatternRewriter &rewriter, Location loc,
                               Value valueToStore, Value base, Value stride,
                               Value vl);

} // namespace intrinsic

} // namespace rvv

std::string getCpuArch();
std::set<std::string> getCpuFeatures();

Value createBroadcast(PatternRewriter &rewriter, Location loc,
                      VectorType vectorTy, Value scalar);

Value createExtuiOrTrunc(PatternRewriter &rewriter, Location loc, Type targetTy,
                         Value val);

Value createMemRefToRawPtr(PatternRewriter &rewriter, Location loc,
                           Value memref);

Value convertToScalableVector(PatternRewriter &rewriter, Location loc,
                              Value fixedVector, VectorType scalableVecTy);
Value convertToFixedVector(PatternRewriter &rewriter, Location loc,
                           Value scalableVector, VectorType fixedVecTy);

Value createI32(PatternRewriter &rewriter, Location loc, int32_t val);
Value createI64(PatternRewriter &rewriter, Location loc, int64_t val);

Value createGep(PatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
                Value index);

} // namespace mlir::triton::cpu