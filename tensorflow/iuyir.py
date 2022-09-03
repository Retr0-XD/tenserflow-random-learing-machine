import tensorflow as tf


named=tf.Variable(([[1,2,3,4],[4,5,6,7]]),shape=(2,4),name='Y',dtype=tf.int32)
print(named.numpy())
print(named.shape())
print(tf.rank(named))
