# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for zero_out ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test


import tensorflow as tf
print(tf.__version__)

try:
  from feature_hash_op.python.ops.feature_hash_ops import feature_hash_op
except ImportError:
  from feature_hash_ops import feature_hash_op


class FeatureHashOpTest(test.TestCase):

  def testFeatureHashOp(self):
    with self.test_session():
      feature_id_hashtable = tf.get_variable(
                "FeatureHashTable", [10, 3],
                initializer=tf.zeros_initializer(dtype=tf.int32),
                dtype=tf.int32, trainable=False)

      features = tf.constant([1,1,1,2,2,2,2,2,2,3,3,3,4,4,4], shape=[5,3], dtype=tf.int32)

      features_ids = feature_hash_op(feature_id_hashtable,features,size_per_bucket=12,size_per_key=10)

      init_op = tf. global_variables_initializer()

      with tf.Session() as sess:
        sess.run(init_op)
        res = sess.run(features_ids)
        print(res)
        self.assertAllClose(
          res, np.array([8,6,6,2,3]))


if __name__ == '__main__':
  test.main()
