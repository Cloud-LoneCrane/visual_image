import tensorflow as tf
import Cifar10
from tensorflow import keras
import numpy as np
import pprint
dtype = tf.float32
High = 32
Width = 32
Channels = 3
Classes = 10

filepath = r"D:\data\cifar-10-python\cifar-10-batches-py"
cifar10 = Cifar10.Cifar10(filepath)

# input = keras.Input((High, Width, Channels))


def load_model():
    x = tf.placeholder(dtype=tf.float32, shape=[None, High, Width, Channels])
    y = tf.placeholder(dtype=tf.float32, shape=[None, Classes])

    activate = tf.nn.selu
    with tf.variable_scope("model"):
        # (32, 32, 3)
        Conv1 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(x)
        # (32, 32, 64)
        Conv2 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(
            Conv1)
        # (32, 32, 64)
        Conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(
            Conv2)
        # (32, 32, 64)->pool2: (16, 16, 128)
        Conv4 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(
            Conv3)
        pool1 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(Conv4)
        # (16, 16, 128)
        Conv5 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(
            pool1)
        # (16, 16, 128)->Pool4: (4, 4, 512)
        Conv6 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(
            Conv5)
        pool2 = keras.layers.MaxPooling2D(pool_size=4, strides=4)(Conv6)
        # (4, 4, 512)
        Conv7 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                    activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(
            pool2)
        # (4, 4, 512)->Pool4：(1, 1, 1024)
        pool3 = keras.layers.MaxPooling2D(pool_size=4, strides=4)(Conv7)
        Flatten = keras.layers.Flatten()(Conv7)
        # (1024,)
        Dense1 = keras.layers.Dense(units=256)
        dense1 = Dense1(Flatten)
        model_out = keras.layers.Dense(Classes)(dense1)

    # 连同图结构一同加载
    ckpt = tf.train.get_checkpoint_state('./logs_tf/')
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        x = tf.get_default_graph().get_tensor_by_name("Placeholder")
        # out = sess.run(model_out, feed_dict={x: cifar10.test_imgs[0:1000]})
        # c = tf.argmax(tf.nn.softmax(out, axis=-1), axis=-1).eval()
        # d = tf.argmax(tf.nn.softmax(tf.cast(cifar10.test_labels[0:1000], dtype=tf.float32), axis=-1), axis=-1).eval()
        # print(tf.reduce_mean(tf.cast(tf.equal(c, d), dtype=tf.float32)).eval())
    return None


def load_model1():
    ckpt = tf.train.get_checkpoint_state('logs_tf2/1/model/')
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        # for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
        #     print(tensor_name)

        x = graph.get_tensor_by_name(name='input_x:0')
        y = graph.get_tensor_by_name(name="input_y:0")
        loss = graph.get_tensor_by_name(name="loss:0")
        accuray = graph.get_tensor_by_name(name="accuracy:0")
        model_out = graph.get_tensor_by_name(name='model/model_out/BiasAdd:0')
        # pprint.pprint([x, y, loss, accuray, model_out])
        #
        out = sess.run(model_out, feed_dict={x: cifar10.test_imgs[0:1000]})
        c = tf.argmax(tf.nn.softmax(out, axis=-1), axis=-1).eval()
        d = tf.argmax(tf.nn.softmax(tf.cast(cifar10.test_labels[0:1000], dtype=tf.float32), axis=-1), axis=-1).eval()
        print(tf.reduce_mean(tf.cast(tf.equal(c, d), dtype=tf.float32)).eval())
        print(sess.run(accuray, feed_dict={x: cifar10.test_imgs[0:1000],
                                           y: cifar10.test_labels[0:1000]}))

    return None

def load_model2():
    # 只加载数据，不加载图结构，可以在新图中改变batch_size等的值
    # 不过需要注意，Saver对象实例化之前需要定义好新的图结构，否则会报错
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model/')
        saver.restore(sess, ckpt.model_checkpoint_path)

    return None


if __name__ == "__main__":
    load_model1()