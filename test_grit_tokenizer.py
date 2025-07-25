from gritlm import GritLM

tokenizer = GritLM("GritLM/GritLM-7B", torch_dtype="auto").tokenizer

diff = """diff --git a/tensorflow/contrib/learn/python/learn/estimators/__init__.py b/tensorflow/contrib/learn/python/learn/estimators/__init__.py
index c8f012877fa968..67fbd6348b81da 100644
--- a/tensorflow/contrib/learn/python/learn/estimators/__init__.py
+++ b/tensorflow/contrib/learn/python/learn/estimators/__init__.py
@@ -42,3 +42,4 @@
 from tensorflow.contrib.learn.python.learn.estimators.rnn import TensorFlowRNNClassifier
 from tensorflow.contrib.learn.python.learn.estimators.rnn import TensorFlowRNNRegressor
 from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig
+from tensorflow.contrib.learn.python.learn.estimators.sdca_optimizer import SDCAOptimizer
diff --git a/tensorflow/contrib/learn/python/learn/estimators/linear.py b/tensorflow/contrib/learn/python/learn/estimators/linear.py
index 12dc1001694046..1013f2748e36d4 100644
--- a/tensorflow/contrib/learn/python/learn/estimators/linear.py
+++ b/tensorflow/contrib/learn/python/learn/estimators/linear.py
@@ -20,9 +20,13 @@
 from __future__ import print_function
 
 from tensorflow.contrib import layers
+from tensorflow.contrib.framework.python.ops import variables as contrib_variables
 from tensorflow.contrib.learn.python.learn.estimators import _sklearn
 from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
+from tensorflow.contrib.learn.python.learn.estimators import sdca_optimizer
 from tensorflow.contrib.learn.python.learn.estimators.base import DeprecatedMixin
+from tensorflow.python.framework import ops
+from tensorflow.python.ops import logging_ops
 
 
 class LinearClassifier(dnn_linear_combined.DNNLinearCombinedClassifier):
@@ -36,9 +40,25 @@ class LinearClassifier(dnn_linear_combined.DNNLinearCombinedClassifier):
     installed_x_impression = crossed_column(
         [installed_app_id, impression_app_id])
 
+    # Estimator using the default optimizer.
     estimator = LinearClassifier(
         feature_columns=[impression_app_id, installed_x_impression])
 
+    # Or estimator using the FTRL optimizer with regularization.
+    estimator = LinearClassifier(
+        feature_columns=[impression_app_id, installed_x_impression],
+        optimizer=tf.train.FtrlOptimizer(
+          learning_rate=0.1,
+          l1_regularization_strength=0.001
+        ))
+
+    # Or estimator using the SDCAOptimizer.
+    estimator = LinearClassifier(
+       feature_columns=[impression_app_id, installed_x_impression],
+       optimizer=tf.contrib.learn.SDCAOptimizer(
+         example_id_column='example_id', symmetric_l2_regularization=2.0
+       ))
+
     # Input builders
     def input_fn_train: # returns x, y
       ...
@@ -71,8 +91,9 @@ def input_fn_eval: # returns x, y
     weight_column_name: A string defining feature column name representing
       weights. It is used to down weight or boost examples during training. It
       will be multiplied by the loss of the example.
-    optimizer: An instance of `tf.Optimizer` used to train the model. If `None`,
-      will use an Ftrl optimizer.
+    optimizer: The optimizer used to train the model. If specified, it should be
+      either an instance of `tf.Optimizer` or the SDCAOptimizer. If `None`, the
+      Ftrl optimizer will be used.
     gradient_clip_norm: A float > 0. If provided, gradients are clipped
       to their global norm with this clipping ratio. See tf.clip_by_global_norm
       for more details.
@@ -100,7 +121,31 @@ def _get_train_ops(self, features, targets):
     """See base class."""
     if self._linear_feature_columns is None:
       self._linear_feature_columns = layers.infer_real_valued_columns(features)
-    return super(LinearClassifier, self)._get_train_ops(features, targets)
+    if not isinstance(self._linear_optimizer, sdca_optimizer.SDCAOptimizer):
+      return super(LinearClassifier, self)._get_train_ops(features, targets)
+
+    # SDCA currently supports binary classification only.
+    if self._n_classes > 2:
+      raise ValueError(
+          "SDCA does not currently support multi-class classification.")
+    global_step = contrib_variables.get_global_step()
+    assert global_step
+
+    logits, columns_to_variables, _ = layers.weighted_sum_from_feature_columns(
+        columns_to_tensors=features,
+        feature_columns=self._linear_feature_columns,
+        num_outputs=self._num_label_columns(),
+        weight_collections=[self._linear_weight_collection],
+        name="linear")
+    with ops.control_dependencies([self._centered_bias()]):
+      loss = self._loss(logits, targets, self._get_weight_tensor(features))
+    logging_ops.scalar_summary("loss", loss)
+
+    train_ops = self._linear_optimizer.get_train_step(
+        self._linear_feature_columns, self._weight_column_name, "logistic_loss",
+        features, targets, columns_to_variables, global_step)
+
+    return train_ops, loss
 
   @property
   def weights_(self):
@@ -175,6 +220,9 @@ def __init__(self,
 
   def _get_train_ops(self, features, targets):
     """See base class."""
+    if isinstance(self._linear_optimizer, sdca_optimizer.SDCAOptimizer):
+      raise ValueError("SDCAOptimizer does not currently support regression.")
+
     if self._linear_feature_columns is None:
       self._linear_feature_columns = layers.infer_real_valued_columns(features)
     return super(LinearRegressor, self)._get_train_ops(features, targets)
diff --git a/tensorflow/contrib/learn/python/learn/estimators/linear_test.py b/tensorflow/contrib/learn/python/learn/estimators/linear_test.py
index 269206628e1d3f..57a36b9f7ac3f2 100644
--- a/tensorflow/contrib/learn/python/learn/estimators/linear_test.py
+++ b/tensorflow/contrib/learn/python/learn/estimators/linear_test.py
@@ -89,6 +89,162 @@ def input_fn():
     loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
     self.assertLess(loss, 0.01)
 
+  def testSdcaOptimizerRealValuedFeatureWithInvalidDimension(self):
+    """Tests a ValueError is raised if a real valued feature has dimension>1."""
+
+    def input_fn():
+      return {
+          'example_id': tf.constant(['1', '2']),
+          'sq_footage': tf.constant([[800.0, 200.0], [650.0, 500.0]])
+      }, tf.constant([[1.0], [0.0]])
+
+    sq_footage = tf.contrib.layers.real_valued_column('sq_footage', dimension=2)
+    sdca_optimizer = tf.contrib.learn.SDCAOptimizer(
+        example_id_column='example_id')
+    classifier = tf.contrib.learn.LinearClassifier(feature_columns=[sq_footage],
+                                                   optimizer=sdca_optimizer)
+    with self.assertRaises(ValueError):
+      _ = classifier.fit(input_fn=input_fn, steps=100)
+
+  def testSdcaOptimizerRealValuedFeatures(self):
+    """Tests LinearClasssifier with SDCAOptimizer and real valued features."""
+
+    def input_fn():
+      return {
+          'example_id': tf.constant(['1', '2']),
+          'maintenance_cost': tf.constant([[500.0], [200.0]]),
+          'sq_footage': tf.constant([[800.0], [600.0]]),
+          'weights': tf.constant([[1.0], [1.0]])
+      }, tf.constant([[0], [1]])
+
+    maintenance_cost = tf.contrib.layers.real_valued_column('maintenance_cost')
+    sq_footage = tf.contrib.layers.real_valued_column('sq_footage')
+    sdca_optimizer = tf.contrib.learn.SDCAOptimizer(
+        example_id_column='example_id')
+    classifier = tf.contrib.learn.LinearClassifier(
+        feature_columns=[maintenance_cost, sq_footage],
+        weight_column_name='weights',
+        optimizer=sdca_optimizer)
+    classifier.fit(input_fn=input_fn, steps=100)
+    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
+    self.assertLess(loss, 0.05)
+
+  def testSdcaOptimizerBucketizedFeatures(self):
+    """Tests LinearClasssifier with SDCAOptimizer and bucketized features."""
+
+    def input_fn():
+      return {
+          'example_id': tf.constant(['1', '2', '3']),
+          'price': tf.constant([[600.0], [1000.0], [400.0]]),
+          'sq_footage': tf.constant([[1000.0], [600.0], [700.0]]),
+          'weights': tf.constant([[1.0], [1.0], [1.0]])
+      }, tf.constant([[1], [0], [1]])
+
+    price_bucket = tf.contrib.layers.bucketized_column(
+        tf.contrib.layers.real_valued_column('price'),
+        boundaries=[500.0, 700.0])
+    sq_footage_bucket = tf.contrib.layers.bucketized_column(
+        tf.contrib.layers.real_valued_column('sq_footage'),
+        boundaries=[650.0])
+    sdca_optimizer = tf.contrib.learn.SDCAOptimizer(
+        example_id_column='example_id',
+        symmetric_l2_regularization=1.0)
+    classifier = tf.contrib.learn.LinearClassifier(
+        feature_columns=[price_bucket, sq_footage_bucket],
+        weight_column_name='weights',
+        optimizer=sdca_optimizer)
+    classifier.fit(input_fn=input_fn, steps=50)
+    scores = classifier.evaluate(input_fn=input_fn, steps=2)
+    self.assertGreater(scores['accuracy'], 0.9)
+
+  def testSdcaOptimizerSparseFeatures(self):
+    """Tests LinearClasssifier with SDCAOptimizer and sparse features."""
+
+    def input_fn():
+      return {
+          'example_id': tf.constant(['1', '2', '3']),
+          'price': tf.constant([[0.4], [0.6], [0.3]]),
+          'country': tf.SparseTensor(values=['IT', 'US', 'GB'],
+                                     indices=[[0, 0], [1, 3], [2, 1]],
+                                     shape=[3, 5]),
+          'weights': tf.constant([[1.0], [1.0], [1.0]])
+      }, tf.constant([[1], [0], [1]])
+
+    price = tf.contrib.layers.real_valued_column('price')
+    country = tf.contrib.layers.sparse_column_with_hash_bucket(
+        'country', hash_bucket_size=5)
+    sdca_optimizer = tf.contrib.learn.SDCAOptimizer(
+        example_id_column='example_id')
+    classifier = tf.contrib.learn.LinearClassifier(
+        feature_columns=[price, country],
+        weight_column_name='weights',
+        optimizer=sdca_optimizer)
+    classifier.fit(input_fn=input_fn, steps=50)
+    scores = classifier.evaluate(input_fn=input_fn, steps=2)
+    self.assertGreater(scores['accuracy'], 0.9)
+
+  def testSdcaOptimizerCrossedFeatures(self):
+    """Tests LinearClasssifier with SDCAOptimizer and crossed features."""
+
+    def input_fn():
+      return {
+          'example_id': tf.constant(['1', '2', '3']),
+          'language': tf.SparseTensor(values=['english', 'italian', 'spanish'],
+                                      indices=[[0, 0], [1, 0], [2, 0]],
+                                      shape=[3, 1]),
+          'country': tf.SparseTensor(values=['US', 'IT', 'MX'],
+                                     indices=[[0, 0], [1, 0], [2, 0]],
+                                     shape=[3, 1])
+      }, tf.constant([[0], [0], [1]])
+
+    language = tf.contrib.layers.sparse_column_with_hash_bucket(
+        'language', hash_bucket_size=5)
+    country = tf.contrib.layers.sparse_column_with_hash_bucket(
+        'country', hash_bucket_size=5)
+    country_language = tf.contrib.layers.crossed_column(
+        [language, country], hash_bucket_size=10)
+    sdca_optimizer = tf.contrib.learn.SDCAOptimizer(
+        example_id_column='example_id')
+    classifier = tf.contrib.learn.LinearClassifier(
+        feature_columns=[country_language],
+        optimizer=sdca_optimizer)
+    classifier.fit(input_fn=input_fn, steps=10)
+    scores = classifier.evaluate(input_fn=input_fn, steps=2)
+    self.assertGreater(scores['accuracy'], 0.9)
+
+  def testSdcaOptimizerMixedFeatures(self):
+    """Tests LinearClasssifier with SDCAOptimizer and a mix of features."""
+
+    def input_fn():
+      return {
+          'example_id': tf.constant(['1', '2', '3']),
+          'price': tf.constant([[0.6], [0.8], [0.3]]),
+          'sq_footage': tf.constant([[900.0], [700.0], [600.0]]),
+          'country': tf.SparseTensor(values=['IT', 'US', 'GB'],
+                                     indices=[[0, 0], [1, 3], [2, 1]],
+                                     shape=[3, 5]),
+          'weights': tf.constant([[3.0], [1.0], [1.0]])
+      }, tf.constant([[1], [0], [1]])
+
+    price = tf.contrib.layers.real_valued_column('price')
+    sq_footage_bucket = tf.contrib.layers.bucketized_column(
+        tf.contrib.layers.real_valued_column('sq_footage'),
+        boundaries=[650.0, 800.0])
+    country = tf.contrib.layers.sparse_column_with_hash_bucket(
+        'country', hash_bucket_size=5)
+    sq_footage_country = tf.contrib.layers.crossed_column(
+        [sq_footage_bucket, country],
+        hash_bucket_size=10)
+    sdca_optimizer = tf.contrib.learn.SDCAOptimizer(
+        example_id_column='example_id')
+    classifier = tf.contrib.learn.LinearClassifier(
+        feature_columns=[price, sq_footage_bucket, country, sq_footage_country],
+        weight_column_name='weights',
+        optimizer=sdca_optimizer)
+    classifier.fit(input_fn=input_fn, steps=50)
+    scores = classifier.evaluate(input_fn=input_fn, steps=2)
+    self.assertGreater(scores['accuracy'], 0.9)
+
   def testEval(self):
     """Tests that eval produces correct metrics.
     """
diff --git a/tensorflow/contrib/learn/python/learn/estimators/sdca_optimizer.py b/tensorflow/contrib/learn/python/learn/estimators/sdca_optimizer.py
new file mode 100644
index 00000000000000..093c5d9875c3fe
--- /dev/null
+++ b/tensorflow/contrib/learn/python/learn/estimators/sdca_optimizer.py
@@ -0,0 +1,157 @@
+"""Linear Estimators."""
+#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
+#
+#  Licensed under the Apache License, Version 2.0 (the "License");
+#  you may not use this file except in compliance with the License.
+#  You may obtain a copy of the License at
+#
+#   http://www.apache.org/licenses/LICENSE-2.0
+#
+#  Unless required by applicable law or agreed to in writing, software
+#  distributed under the License is distributed on an "AS IS" BASIS,
+#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+#  See the License for the specific language governing permissions and
+#  limitations under the License.
+from __future__ import absolute_import
+from __future__ import division
+from __future__ import print_function
+
+import uuid
+
+from tensorflow.contrib import layers
+from tensorflow.contrib.linear_optimizer.python.ops import sdca_ops
+from tensorflow.python.framework import ops
+from tensorflow.python.ops import array_ops
+from tensorflow.python.ops import math_ops
+from tensorflow.python.ops import sparse_ops
+
+
+class SDCAOptimizer(object):
+  """Wrapper class for SDCA optimizer.
+
+  Example usage:
+    real_feature_column = real_valued_column(...)
+    sparse_feature_column = sparse_column_with_hash_bucket(...)
+    sdca_optimizer = linear.SDCAOptimizer(example_id_column='example_id',
+                                          symmetric_l2_regularization=2.0)
+    classifier = linear.LinearClassifier(
+        feature_columns=[real_feature_column, sparse_feature_column],
+        weight_column_name=...,
+        optimizer=sdca_optimizer)
+    classifier.train(input_fn_train, steps=50)
+    classifier.evaluate(input_fn=input_fn_eval)
+
+  Here the expectation is that the input_fn_* functions passed to train and
+  evaluate return a pair of (dict, label_tensor) where dict has an "example_id"
+  key whose value is a tensor of shape [batch_size] and dtype string.
+  """
+
+  def __init__(self,
+               example_id_column,
+               symmetric_l1_regularization=0.0,
+               symmetric_l2_regularization=1.0):
+    self._example_id_column = example_id_column
+    self._symmetric_l1_regularization = symmetric_l1_regularization
+    self._symmetric_l2_regularization = symmetric_l2_regularization
+
+  def get_train_step(self, linear_feature_columns, weight_column_name,
+                     loss_type, features, targets, columns_to_variables,
+                     global_step):
+    """Returns the training operation of an SDCAModel optimizer."""
+
+    # TODO(sibyl-vie3Poto): Rename this method to convert_to_sparse_tensor and move under
+    # contrib/framework.
+    def _dense_to_sparse_tensor(dense_tensor):
+      """Returns a SparseTensor for the input dense_tensor."""
+      ignore_value = 0.0
+      sparse_indices = array_ops.where(math_ops.not_equal(
+          dense_tensor, math_ops.cast(ignore_value, dense_tensor.dtype)))
+      sparse_values = array_ops.gather_nd(dense_tensor, sparse_indices)
+      # SparseTensor needs the shape to be converted to int64.
+      int64_shape = math_ops.to_int64(array_ops.shape(dense_tensor))
+      return ops.SparseTensor(sparse_indices, sparse_values, shape=int64_shape)
+
+    def _training_examples_and_variables():
+      """Returns dictionaries for training examples and variables."""
+      batch_size = targets.get_shape()[0]
+
+      # Iterate over all feature columns and create appropriate lists for dense
+      # and sparse features as well as dense and sparse weights (variables) for
+      # SDCA.
+      # TODO(sibyl-vie3Poto): Reshape variables stored as values in column_to_variables
+      # dict as 1-dimensional tensors.
+      dense_features, sparse_features = [], []
+      dense_features_weights, sparse_features_weights = [], []
+      for column in sorted(set(linear_feature_columns), key=lambda x: x.key):
+        transformed_tensor = features[column]
+        if isinstance(column, layers.feature_column.
+                      _RealValuedColumn):  # pylint: disable=protected-access
+          # A real-valued column corresponds to a dense feature in SDCA.
+          if column.dimension != 1:
+            raise ValueError(
+                "Invalid column dimension %d for column %s. SDCAOptimizer "
+                "supports only 1-dimensional dense feature columns." %
+                (column.dimension, column.name))
+
+          dense_features.append(array_ops.reshape(transformed_tensor,
+                                                  shape=[-1]))
+          # For real valued columns, the variables list contains exactly one
+          # element.
+          dense_features_weights.append(columns_to_variables[column][0])
+        elif isinstance(column, layers.feature_column.
+                        _BucketizedColumn):  # pylint: disable=protected-access
+          # A bucketized column corresponds to a sparse feature in SDCA. The
+          # bucketized feature is "sparsified" for SDCA by converting it to a
+          # SparseTensor respresenting the one-hot encoding of the bucketized
+          # feature.
+          dense_bucket_tensor = column.to_dnn_input_layer(transformed_tensor)
+          sparse_bucket_tensor = _dense_to_sparse_tensor(dense_bucket_tensor)
+          sparse_features.append(sparse_bucket_tensor)
+          # For bucketized columns, the variables list contains exactly one
+          # element.
+          sparse_features_weights.append(columns_to_variables[column][0])
+        elif isinstance(column,
+                        (layers.feature_column.
+                         _CrossedColumn,  # pylint: disable=protected-access
+                         layers.feature_column._SparseColumn
+                        )):  # pylint: disable=protected-access
+          weights_tensor = ops.SparseTensor(
+              indices=transformed_tensor.indices,
+              values=array_ops.ones_like(transformed_tensor.values),
+              shape=transformed_tensor.shape)
+          sparse_features_tensor = sparse_ops.sparse_merge(transformed_tensor,
+                                                           weights_tensor,
+                                                           column.length)
+          sparse_features.append(math_ops.to_float(sparse_features_tensor))
+          sparse_features_weights.append(columns_to_variables[column][0])
+        else:
+          raise ValueError("SDCAOptimizer does not support column type %s." %
+                           type(column).__name__)
+
+      example_weights = array_ops.reshape(
+          features[weight_column_name],
+          shape=[-1]) if weight_column_name else array_ops.ones([batch_size])
+      example_ids = features[self._example_id_column]
+      examples = dict(
+          sparse_features=sparse_features,
+          dense_features=dense_features,
+          example_labels=math_ops.to_float(
+              array_ops.reshape(targets, shape=[-1])),
+          example_weights=example_weights,
+          example_ids=example_ids)
+      sdca_variables = dict(sparse_features_weights=sparse_features_weights,
+                            dense_features_weights=dense_features_weights)
+      return examples, sdca_variables
+
+    options = dict(
+        symmetric_l1_regularization=self._symmetric_l1_regularization,
+        symmetric_l2_regularization=self._symmetric_l2_regularization,
+        loss_type=loss_type)
+    training_examples, training_variables = _training_examples_and_variables()
+    # TODO(sibyl-vie3Poto): Take care of cleanup, when the API to reset the container is
+    # available.
+    sdca_model = sdca_ops.SdcaModel(container=uuid.uuid4().hex,
+                                    examples=training_examples,
+                                    variables=training_variables,
+                                    options=options)
+    return sdca_model.minimize(global_step=global_step)"""

print(len(tokenizer.tokenize(diff)), flush=True)
