from keras import backend as K
import keras
import numpy as np
import math
import os
import glob
from keras.callbacks import Callback
import tensorflow as tf
import cv2


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width,
                            colorspace=channel, encoded_image_string=image_string)


class MyTensorBoard(keras.callbacks.Callback):
    def __init__(self, log_dir='./logs', batch_size=64, update_features_freq=1, input_images=None,
                 write_graph=True, write_features=False, update_freq='epoch'):
        # 1.初始化父类
        super(MyTensorBoard, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        self.input_images = input_images[0]     # 只取输入图像的第一个batch？
        self.log_dir = log_dir
        self.merged = None
        self.im_summary = []    # image的summary
        self.lr_summary = None  # lr的summary
        self.write_graph = write_graph
        self.write_features = write_features
        self.batch_size = batch_size
        self.update_features_freq = update_features_freq
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq

        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def set_model(self, model):
        # set_model会被自动调用一次，在set_model中定义要收集的内容，summary
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.merged is None:
            # 显示特征图
            # 遍历所有的网络层
            for layer in self.model.layers:
                # 获取当前层的输出与名称
                # feature_map = layer.output
                feature_map = layer.get_output_at(0)
                feature_map_name = layer.name.replace(':', '_')

                if self.write_features and len(K.int_shape(feature_map)) == 4:  # 如果feature_map的维度是4维的，那么就画图
                    # 展开特征图并拼接成大图
                    flat_concat_feature_map = self._concact_features(feature_map)
                    # 判断展开的特征图最后通道数是否是1
                    shape = K.int_shape(flat_concat_feature_map)
                    assert len(shape) == 4 and shape[-1] == 1
                    # 写入tensorboard
                    self.im_summary.append(tf.summary.image(feature_map_name, flat_concat_feature_map, 4))
                    # 第三个参数为tensorboard展示几个

            # 显示学习率的变化
            # self.lr_summary = tf.summary.scalar("learning_rate", self.model.optimizer.lr)

        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    # # Not Change
    # def set_params(self, params):
    #     self.params = params

    # # Not implement
    # def on_train_begin(self, logs=None):
    #     """Called at the beginning of training.
    #
    #     Subclasses should override for any actions to run.
    #
    #     Arguments:
    #         logs: dict. Currently no data is passed to this argument for this method
    #           but that may change in the future.
    #     """

    def on_train_end(self, logs=None):
        self.writer.close()

    # # Not implement
    # def on_epoch_begin(self, epoch, logs=None):
    #     """Called at the start of an epoch.
    #
    #     Subclasses should override for any actions to run. This function should only
    #     be called during TRAIN mode.
    #
    #     Arguments:
    #         epoch: integer, index of epoch.
    #         logs: dict. Currently no data is passed to this argument for this method
    #           but that may change in the future.
    #     """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data:
            val_data = self.validation_data
            tensors = (self.model.inputs + self.model.targets + self.model.sample_weights)

            if self.model.uses_learning_phase:
                tensors += [K.learning_phase()]

            assert len(val_data) == len(tensors)
            val_size = val_data[0].shape[0]
            i = 0
            while i < val_size:
                step = min(self.batch_size, val_size - i)
                if self.model.uses_learning_phase:
                    # do not slice the learning phase
                    batch_val = [x[i:i + step] for x in val_data[:-1]]
                    batch_val.append(val_data[-1])
                else:
                    batch_val = [x[i:i + step] for x in val_data]
                assert len(batch_val) == len(tensors)
                feed_dict = dict(zip(tensors, batch_val))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)
                i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    # # Not implement
    # def on_batch_begin(self, batch, logs=None):
    #     """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen

        # 每update_features_freq个batch刷新特征图
        if batch % self.update_features_freq == 0:
            # 计算summary_image
            feed_dict = dict(zip(self.model.inputs, self.input_images[np.newaxis, ...]))
            for i in range(len(self.im_summary)):
                summary = self.sess.run(self.im_summary[i], feed_dict)
                self.writer.add_summary(summary, self.samples_seen)

        # # 每个batch显示学习率
        # summary = self.sess.run(self.lr_summary, {self.model.optimizer.lr: K.eval(self.model.optimizer.lr)})
        # self.writer.add_summary(summary, self.samples_seen)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def _concact_features(self, conv_output):
        """
        对特征图进行reshape拼接
        :param conv_output:输入多通道的特征图
        :return: all_concact
        """
        all_concact = None

        num_or_size_splits = conv_output.get_shape().as_list()[-1]  # 就说channel数
        each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
        # each_convs就是按通道分离出来的特征图，这里就是一个通道一个通道分离出来

        if num_or_size_splits < 4:
            # 对于特征图少于4通道的认为是输入，直接横向concact输出即可
            concact_size = num_or_size_splits   # 通道数
            all_concact = each_convs[0]
            for i in range(concact_size - 1):
                all_concact = tf.concat([all_concact, each_convs[i + 1]], 1)
        else:
            # 对于特征图，则拼接成正方形
            concact_size = int(math.sqrt(num_or_size_splits) / 1)
            for i in range(concact_size):
                row_concact = each_convs[i * concact_size]
                for j in range(concact_size - 1):
                    row_concact = tf.concat(
                        [row_concact, each_convs[i * concact_size + j + 1]], 1)
                if i == 0:
                    all_concact = row_concact
                else:
                    all_concact = tf.concat([all_concact, row_concact], 2)
        return all_concact

