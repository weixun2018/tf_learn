import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
    a = tf.Variable(0, name='a')
    print(a.graph)
    assert a.graph == g1

with tf.Graph().as_default() as g2:
    b = tf.Variable(1, name='b')
    print(b.graph)
    assert b.graph == g2
