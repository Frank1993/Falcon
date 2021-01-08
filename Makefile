CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

FEATURE_HASH_SRCS =  $(wildcard feature_hash_op/cc/kernels/*.cc) $(wildcard feature_hash_op/cc/kernels/*.h) $(wildcard feature_hash_op/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

FEATURE_HASH_TARGET_LIB = feature_hash_op/python/ops/_feature_hash_ops.so

# zero_out op for CPU
feature_hash_op: $(FEATURE_HASH_TARGET_LIB)

$(FEATURE_HASH_TARGET_LIB): $(FEATURE_HASH_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

feature_hash_test: feature_hash_op/python/ops/feature_hash_ops_test.py feature_hash_op/python/ops/feature_hash_ops.py $(FEATURE_HASH_TARGET_LIB)
	$(PYTHON_BIN_PATH) feature_hash_op/python/ops/feature_hash_ops_test.py

feature_hash_pip_pkg: $(FEATURE_HASH_TARGET_LIB)
	./build_pip_pkg.sh make artifacts
