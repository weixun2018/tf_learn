import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.001
max_train_steps = 1000

# train_X = np.random.randint(50, size=(20,1))
# train_X = np.reshape(np.array(list(map(float, train_X))), [20, 1])
train_X = np.reshape(np.array([i for i in range(10)], dtype=np.float32), [10, 1])
temp = np.random.rand(10, 1)
# train_X += temp
print(train_X.dtype, train_X.shape)
# train_Y = np.random.randint(50, size=(20,1))
# train_Y = np.reshape(np.array(list(map(float, train_Y))), [20, 1])
train_Y = np.reshape(np.array([i * 2 for i in range(10)], dtype=np.float32), [10, 1])
# train_Y += np.random.rand(10, 1) * 2
print(train_Y.dtype, train_Y.shape)

X = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.random_normal([1, 1], name='weights'))
b = tf.Variable(tf.zeros([1]), name='bias')
Y = tf.matmul(X, w) + b
Y_ = tf.placeholder(tf.float32, [None, 1])

loss = tf.reduce_mean(tf.pow(Y - Y_, 2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('=== start training ===')
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})
        if step % 100 == 0:
            print('step:{:4d}, loss:{:.4f}'.format(step, loss.eval(feed_dict={X: train_X, Y_: train_Y})))

    final_loss = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
    print('final loss:{:.4f}'.format(final_loss))

    weight, bias = sess.run([w, b])

    print('Linear Regression model:Y={}*X+{}'.format(weight, bias))

    plt.plot(train_X, train_Y, 'ro', label='Training data')
    plt.plot(train_X, weight*train_X+bias, label='Fitted Line')
    plt.legend()
    plt.show()
