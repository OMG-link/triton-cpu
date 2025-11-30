# build on k1
```
source triton-venv/bin/activate

git submodule update --init --recursive

TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_OFFLINE_BUILD=1 \
JSON_SYSPATH="/home/shenrh/.triton/json" \
LLVM_BUILD_DIR=/opt/llvm/llvm21 \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
MAX_JOBS=1 \
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
pip install -e python --no-build-isolation -vvv 
```

baseline-test: change the compiler 
```
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_OFFLINE_BUILD=1 \
JSON_SYSPATH="/home/shenrh/.triton/json" \
LLVM_BUILD_DIR=/opt/llvm/llvm21-raw \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
MAX_JOBS=7 \
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
pip install -e python --no-build-isolation -vvv
```

# run demos on k1
```
source triton-venv/bin/activate 

CC=/opt/llvm/llvm20/bin/clang TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./python/tutorials/ir-dump TRITON_CPU_BACKEND=1 python3 python/tutorials/cpu-blocked-matmul.py 
```

# build on macos
rm -rf python/build/ python/triton.egg-info/
find python -name "*.so" -delete
find python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

TRITON_BUILD_WITH_CLANG_LLD=true \                                   
TRITON_OFFLINE_BUILD=1 \
JSON_SYSPATH="/home/shenrh/.triton/json" \
LLVM_BUILD_DIR=/opt/llvm/llvm21 \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
MAX_JOBS=6 \
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
SETUPTOOLS_BUILD_FORCE_RECONFIGURE=1 \
SOURCE_DATE_EPOCH=$(date +%s) \
pip install --force-reinstall --no-deps --no-cache-dir -e python --no-build-isolation -vvv


TRITON_LOCAL_LIBOMP_PATH=/opt/homebrew/opt/libomp/ \
MAX_JOBS=6 \
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
pip install -e python --no-build-isolation 

# install python packages
- 进迭时空预编译的一些 Python 库，可访问网站查找 package 并安装： ·https://git.spacemit.com/archive/pypi/-/packages/· 
```
pip install numpy --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install torch --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install matplotlib --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install pandas --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```
