import tensorflow as tf
import numpy as np

# TFRecordsファイルを保存したパス
filepath = "/home/wangzq/workspace/dataset/ImageNet1k256/TFRecords/train_0001_13485.tfrec"

raw_dataset = tf.data.TFRecordDataset(filepath)

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)