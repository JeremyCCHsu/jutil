import tensorflow as tf
from tensorflow.contrib import keras


def mnist_batcher_in_tanh_vector(
    batch_size,
    capacity=256,
    min_after_dequeue=128,
    ):
    (x, y), (_, _) = keras.datasets.mnist.load_data()
    x = tf.constant(x)
    x = tf.cast(x, tf.float32)
    x = keras.layers.Flatten()(x) / 127.5 - 1.
    y = tf.cast(y, tf.int64)

    return tf.train.shuffle_batch(
        [x, y],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True
    )
