import tensorflow as tf 

a=tf.constant(2,dtype=tf.int32)
b=tf.constant(5,dtype=tf.int32)
c=tf.constant(a+b,dtype=tf.int32)
#c=tf.sum(a,b)

print("the add value of the two modules is :",c.numpy())
