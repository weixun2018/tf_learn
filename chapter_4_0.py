import tensorflow as tf

w = tf.Variable(tf.random_normal(shape=[4, 1], mean=10, stddev=5), name='weights')

w_temp = tf.Variable(w.initial_value, name='weights_temp')

w_twice = tf.Variable(w.initial_value * 2, name='weights_twice')

w_try = tf.Variable(w, name='w_try')

w_list = tf.Variable([1, 2, 3], name='w_list')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))
    print(sess.run(w_temp))
    print(sess.run(w_twice))
    print(sess.run(w_try))
    print(sess.run(w_list))
