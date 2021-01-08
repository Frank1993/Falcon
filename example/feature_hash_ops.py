
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
import tensorflow as tf

feature_hash_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('../feature_hash_op/python/ops/_feature_hash_ops.so'))

def hash_feature_to_id(feature_hash_table, features, size_per_bucket, size_per_key):
    return feature_hash_ops.feature_hash_op(feature_hash_table, features, size_per_bucket=size_per_bucket, size_per_key=size_per_key)

def get_size_per_bucket(size_per_key):
    if size_per_key % 4 == 0:
        size_per_bucket = size_per_key +  4
    else:
        size_per_bucket = size_per_key / 4 * 4 + 4

    return size_per_bucket  

def initialize_feature_id_hashtable(num_buckets, size_per_key):

    size_per_bucket = get_size_per_bucket(size_per_key)

    feature_hashtable = tf.get_variable(
                "FeatureHashTable", [num_buckets, size_per_bucket/4],
                initializer=tf.zeros_initializer(dtype=tf.int32),
                dtype=tf.int32, trainable=False)

    return feature_hashtable