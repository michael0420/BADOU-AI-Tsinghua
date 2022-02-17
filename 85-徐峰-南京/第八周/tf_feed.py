import tensorflow as tf


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

out = tf.multiply(input1, input2)


#fetch

input3 = tf.constant(3.0)
input4 = tf.constant(2.0)
input5 = tf.constant(5.0)

intermed = tf.add(input3, input4)
mul = tf.multiply(input3, intermed)

with tf.Session() as sess:
    print(sess.run([out], feed_dict={input1:[7.], input2:[2.]}))
    res = sess.run([mul, intermed])
    print(res)