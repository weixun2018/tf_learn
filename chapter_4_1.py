import tensorflow as tf

'''
with tf.Session() as sess:
    with tf.variable_scope('foo', initializer=tf.constant_initializer(3)):
        v = tf.get_variable('v', [1])
        print(v.name)
        # assert v.eval() == 3
        sess.run(tf.global_variables_initializer())
        weight = tf.get_variable('weight', [1], initializer=tf.constant_initializer(6))
        sess.run(tf.global_variables_initializer())
        print(sess.run(weight))
        assert weight.eval() == 6
        with tf.variable_scope('bar'):
            v = tf.get_variable('v', [1])
            v.initializer.run()
            print(v.eval())
            assert v.eval() == 3
        with tf.variable_scope('temp', initializer=tf.constant_initializer(8)):
            v = tf.get_variable('v', [1])
            v.initializer.run()
            print(v.eval())
            assert v.eval() == 8
        with tf.variable_scope('bar', reuse=True):
            v = tf.get_variable('v', [1])
            print(v.eval())

    # v = tf.get_variable('v', [1], initializer=tf.constant_initializer(3))
    # print(v.name)
    # # v.initializer.run()
    # sess.run(tf.global_variables_initializer())
    # print(v.eval())
 '''

# 注意， bias1 的定义方式
with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2, 3])
    #bias1 = tf.Variable([0.52], name='bias')

print(Weights1.name)
#print(bias1.name)

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')
    bias2 = tf.Variable([0.52], name='bias')

print(Weights2.name)
print(bias2.name)

"""
output:

v_scope/Weights:0
v_scope/bias:0
v_scope/Weights:0
v_scope_1/bias:0
"""

"""
test get_variable for reuse
"""
with tf.variable_scope('v_scope_1') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2, 3])
    bias1 = tf.Variable([0.52], name='bias')

print(Weights1.name)
print(bias1.name)

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope_1', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')
    bias2 = tf.get_variable('bias', [1])  # ‘bias

print(Weights2.name)
print(bias2.name)

"""
output:

ValueError: Variable v_scope/bias does not exist, or was not created with tf.get_variable()
v_scope/Weights:0
v_scope/bias:0
"""
