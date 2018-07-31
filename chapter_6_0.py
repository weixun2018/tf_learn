import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], 'X-input')
    y_ = tf.placeholder(tf.float32, [None, 10], 'Y-input')

with tf.name_scope('softmax_layer'):
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.random_normal([784, 10]))
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.random_normal([10]))
    with tf.name_scope('wx_plus_b'):
        y = tf.add(tf.matmul(x, weights), bias)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter('./summary/mnist', sess.graph)

merged = tf.summary.merge_all()

tf.global_variables_initializer().run()

for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})

    if i % 1000 == 0:
        acc, loss_, summary = sess.run([accuracy, loss, merged],
                                       feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
        print('Step:{:5d}  validation accuracy:{:.4f}  loss:{:.1f}'.format(i, acc, loss_))
        # saver.save(sess, 'checkpoints/mnist/step-{}_acc-{}.ckpt'.format(i, acc))
        writer.add_summary(summary, i)

print('Finally, test acc:{:.4f}'.format(
    sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
