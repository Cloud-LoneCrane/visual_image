import tensorflow as tf
from tensorflow import keras
import MyTensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import math

import Cifar10
dtype = tf.float32
High = 32
Width = 32
Channels = 3
Classes = 10

filepath = r"D:\data\cifar-10-python\cifar-10-batches-py"
cifar10 = Cifar10.Cifar10(filepath)

# input = keras.Input((High, Width, Channels))
x = tf.placeholder(dtype=tf.float32, shape=[None, High, Width, Channels], name="input_x")
y = tf.placeholder(dtype=tf.float32, shape=[None, Classes], name="input_y")

activate = tf.nn.selu

with tf.variable_scope("model"):
    # (32, 32, 3)
    Conv1 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(x)
    # (32, 32, 64)
    Conv2 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(Conv1)
    # (32, 32, 64)
    Conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(Conv2)
    # (32, 32, 64)->pool2: (16, 16, 128)
    Conv4 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(Conv3)
    pool1 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(Conv4)
    # (16, 16, 128)
    Conv5 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(pool1)
    # (16, 16, 128)->Pool4: (4, 4, 512)
    Conv6 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(Conv5)
    pool2 = keras.layers.MaxPooling2D(pool_size=4, strides=4)(Conv6)
    # (4, 4, 512)
    Conv7 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                                activation=activate, kernel_initializer=tf.contrib.layers.xavier_initializer())(pool2)
    # (4, 4, 512)->Pool4：(1, 1, 1024)
    pool3 = keras.layers.MaxPooling2D(pool_size=4, strides=4)(Conv7)
    Flatten = keras.layers.Flatten()(Conv7)
    # (1024,)
    Dense1 = keras.layers.Dense(units=256)
    dense1 = Dense1(Flatten)
    model_out = keras.layers.Dense(Classes, name="model_out")(dense1)
    # model_out = Dense2(dense1, name="model_out")


def keras_model():
    model = keras.Model(inputs=input, outputs=model_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['crossentropy'])

    model.summary()
    # TODO 添加数据输入
    filename = "model_tensorboard_img/{epoch:03d}-{val_loss:.5f}.h5"

    tensorboad = keras.callbacks.TensorBoard(log_dir='./logs',
                                             histogram_freq=1, batch_size=16,
                                             write_graph=True, write_grads=False,
                                             write_images=True, embeddings_freq=0, update_freq=500
                                             )

    train_data, train_labels = cifar10.test_imgs, cifar10.test_labels
    model.fit(train_data, train_labels, batch_size=64, epochs=5, verbose=1,
              callbacks=[tensorboad])

    model.save('my_model_last.h5')
    return None


def conver_feature(feature):
    """
    对特征图进行reshape拼接
    :param
    feature: 输入多通道的特征图
    :return: all_concact
    """
    all_concact = None

    num_or_size_splits = feature.get_shape().as_list()[-1]  # 就是channel数
    each_convs = tf.split(feature, num_or_size_splits=num_or_size_splits, axis=3)
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


def tf_model():
    # 定义loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model_out), axis=0,
                          name="loss")

    # 定义optimizer
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 定义准确度
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(model_out, axis=1)), dtype=tf.float32),
                              name="accuracy")

    # summary
    with tf.name_scope("summary"):
        # dense1_w = tf.summary.histogram("dense1_w", Dense1.weights)
        dense1_b = tf.summary.histogram("dense1_b", Dense1.bias)
        summary_loss = tf.summary.scalar("loss", loss)
        summary_accuracy = tf.summary.scalar("accuracy", accuracy)

        # 添加可视化image
        # summary写入image的要求：
        # 只能写入channel=1的feature或者channel=3的RGB彩色图！
        summary_image = tf.summary.image("input_image", x[:3])
        summary_conv1 = tf.summary.image("conv1_feature_map", conver_feature(Conv1))
        summary_conv4 = tf.summary.image("conv4_feature_map", conver_feature(Conv4))
        summary = tf.summary.merge_all()
        # summary = tf.summary.merge([dense1_w, dense1_b, summary_loss, summary_accuracy])

    with tf.name_scope("save_model"):
        saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_write = tf.summary.FileWriter("logs_tf2/4", sess.graph)  # out_dir 为输出路径

        # test_x, test_y = cifar10.test_imgs, cifar10.test_labels
        for step in range(1000):
            print(step)
            train_x, train_y = cifar10.next_batch(128)
            _, summary_out = sess.run([optimizer, summary], feed_dict={x: train_x, y: train_y})

            summary_write.add_summary(summary_out, step)  # 每次循环都要执行一次写入到文件
            if step % 100 == 0:
                saver.save(sess, "logs_tf2/4/model/model", step)
        summary_write.close()

        # 训练完成保存模型和参数
        
    return None


if __name__ == "__main__":
    tf_model()



