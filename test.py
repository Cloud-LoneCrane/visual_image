import tensorflow as tf
import numpy as np


def test_reduce_mean():
    """
    test tf.reduce_mean()
    """
    a = np.array([[2, 4],
                  [8, 10]])

    out = tf.reduce_mean(a, axis=0)

    with tf.Session() as sess:
        print(out.eval())

    return None


def test_argmax():
    "test tf.argmax axis"
    a = np.array([[0, 1, 0, 0],
                  [1, 0, 0, 0]])

    b = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0]])
    out = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a, axis=1), tf.argmax(b, axis=1)), dtype=tf.float32))

    with tf.Session() as sess:
        print(out.eval())
    return None


if __name__ == "__main__":
    # test_reduce_mean()
    # test_argmax()
    print(1000/10)