add_triton_library(TritonCPUToLLVM
    AtomicOpsToLLVM.cpp
    DebugOpsToLLVM.cpp
    UkernelOpsToOneDNNLLVM.cpp
    UkernelOpsToXSMMLLVM.cpp
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    LowerMultiReduction.cpp
    LowerTransposedTransferOp.cpp
    MathToVecLib.cpp
    MemoryOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp

    DEPENDS
    TritonCPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
