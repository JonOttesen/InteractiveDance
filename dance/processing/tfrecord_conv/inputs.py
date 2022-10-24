# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data provider."""
import functools
from input_util import *
import tensorflow as tf


def create_input():
  """Create batched input data.

  Args:
    train_eval_config: A train or eval config.
    dataset_config: A dataset config.
    num_cpu_threads: Number of cpu threads for dataset reading.
    is_training: Whether this is training stage.
  Returns:
    ds: A tf.data.Dataset, with the following features:
      features_{audio, motion}, masked_features_{audio, motion},
      masked_positions_{audio, motion}, mask_{audio, motion}.
  """
  batch_size = 1
  data_files = tf.io.gfile.glob("/home/jon/Documents/dance/tf_sstables/aist_generation_train_v2_tfrecord*")

  name_to_features = {}
  modality_to_params = ["motion", "audio"]

  for modality in ["motion", "audio"]:
    name_to_features.update({
    f"{modality}_sequence": tf.io.VarLenFeature(tf.float32),
    f"{modality}_sequence_shape": tf.io.FixedLenFeature([2], tf.int64),
    f"{modality}_name": tf.io.FixedLenFeature([], tf.string),
    })

  ds = tf.data.TFRecordDataset(data_files)

  # Function to decode a record
  def _decode_and_reshape_record(record):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.dtypes.cast(t, tf.int32)
      example[name] = t

    # Sparse to dense
    for modality in modality_to_params:
      example[f"{modality}_sequence"] = tf.reshape(
          tf.sparse.to_dense(example[f"{modality}_sequence"]),
          example[f"{modality}_sequence_shape"])
    return example

  ds = ds.map(_decode_and_reshape_record, num_parallel_calls=1)

  ds = ds.map(
    functools.partial(
    fact_preprocessing,
    modality_to_params=modality_to_params,
    is_training=False),
    num_parallel_calls=1)

  # We must `drop_remainder` on training because the TPU requires fixed
  # size dimensions.
  # If not using TPU, we *don't* want to drop the remainder when eval.

  ds = ds.batch(batch_size, drop_remainder=False)
  ds = ds.prefetch(1)
  return ds

ds = create_input()

for i in ds:
    print(i.keys())
    print(i["audio_name"])
    print(i["target"].shape)
    print(i["motion_input"].shape)
    print(i["audio_input"].shape)
    exit()