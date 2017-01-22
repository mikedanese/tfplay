
import os.path
import urllib
import logging

import tensorflow as tf
import numpy as np
import pandas as pd


DATA_DIR="/tmp"
DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
TRAIN_STEPS=10000
SEED=5738
RNG=np.random.RandomState(seed=SEED)

CATEGORICAL_COLUMNS=["num_rings"]
CONTINUOUS_COLUMNS=["length","diameter","height","whole_weight","shucked_weight","viscera_weight", "shell_weight"]
LABEL_COLUMN=["sex"]
COLUMNS=LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

def sex_converter(sex):
  return {
      'M': 0,
      'F': 1,
      'I': 2,
  }[sex]


COLUMNS_CONVERTERS={
    "sex": sex_converter,
}

def model(model_type):
  """Logistic regression with FTRL optimizer"""
  wide_columns = [
      tf.contrib.layers.real_valued_column("length"),
      tf.contrib.layers.real_valued_column("diameter"),
      tf.contrib.layers.real_valued_column("height"),
      tf.contrib.layers.real_valued_column("whole_weight"),
      tf.contrib.layers.real_valued_column("shucked_weight"),
      tf.contrib.layers.real_valued_column("viscera_weight"),
      tf.contrib.layers.sparse_column_with_integerized_feature("num_rings", bucket_size=50),
  ]
  deep_columns = [
      tf.contrib.layers.real_valued_column("length"),
      tf.contrib.layers.real_valued_column("diameter"),
      tf.contrib.layers.real_valued_column("height"),
      tf.contrib.layers.real_valued_column("whole_weight"),
      tf.contrib.layers.real_valued_column("shucked_weight"),
      tf.contrib.layers.real_valued_column("viscera_weight"),
      tf.contrib.layers.embedding_column(
				tf.contrib.layers.sparse_column_with_integerized_feature("num_rings", bucket_size=50),
				dimension=8),
  ]
  optimizer = tf.train.FtrlOptimizer(
    learning_rate=0.01,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0)
  return {
		'w': tf.contrib.learn.LinearClassifier(feature_columns=wide_columns, optimizer=optimizer),
		'd': tf.contrib.learn.DNNClassifier(feature_columns=deep_columns, hidden_units=[100,100,50,50],optimizer=optimizer),
		'dl': tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        linear_optimizer=optimizer,
        dnn_feature_columns=deep_columns,
        dnn_optimizer=optimizer,
        dnn_hidden_units=[100, 100, 50, 50]),
  }[model_type]

def input_fn(df):
  """Input builder function."""
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}

  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)

  label = tf.constant(df[LABEL_COLUMN].values)

  return feature_cols, label

def maybe_download(url):
  fname = os.path.join(DATA_DIR, url.split('/')[-1])
  if os.path.isfile(fname):
    return fname
  if os.path.exists(fname):
    raise Error("file exists but is a directory")
  urllib.request.urlretrieve(url, fname)
  return fname

def main():
  logging.getLogger().setLevel(logging.INFO)
  df_data = pd.read_csv(
       tf.gfile.Open(maybe_download(DATA_URL)),
       names=COLUMNS,
       converters=COLUMNS_CONVERTERS,
       skipinitialspace=True,
       engine="python",
  ) 

  split_index = int(len(df_data)*3/4)

  df_train = df_data[split_index:]
  df_test = df_data[:split_index]

  g = tf.Graph()
  with g.as_default() as g:
    m = model('dl')
    init = tf.global_variables_initializer()

    with tf.Session( graph = g ) as s:
      m.fit(input_fn=lambda: input_fn(df_train), steps=TRAIN_STEPS)
      print("training")
      results = m.evaluate(input_fn=lambda: input_fn(df_train), steps=1)
      for key in sorted(results):
        print("%s: %s" % (key, results[key]))
      print("test")
      results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
      for key in sorted(results):
        print("%s: %s" % (key, results[key]))
  tf.reset_default_graph()


if __name__ == "__main__":
  main()

