# import tensorflow as tf

# def focal_loss(gamma=2.0, alpha=0.25):
#     def focal_loss_fixed(y_true, y_pred):
#         y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
#         y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) - tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
#     return focal_loss_fixed