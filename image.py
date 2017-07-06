import tensorflow as tf


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def make_png_thumbnail(x, n):
    '''
    Input:
        `x`: Tensor, value range=[-1, 1), shape=[n*n, h, w, c]
        `n`: sqrt of the number of images
    
    Return:
        `tf.string` (bytes) of the PNG. 
        (write these binary directly into a file)
    '''
    with tf.name_scope('MakeThumbnail'):
        _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, [n, n, h, w, c])
        x = tf.transpose(x, [0, 2, 1, 3, 4])
        x = tf.reshape(x, [n * h, n * w, c])
        x = x / 2. + .5
        x = tf.image.convert_image_dtype(x, tf.uint8, saturate=True)
        x = tf.image.encode_png(x)
    return x
