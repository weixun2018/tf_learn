import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags

flags.DEFINE_string('dataset_dir', 'data/mnist_data', 'directory for storing mnist data')
flags.DEFINE_float('learning_rate', 0.5, 'learning rate')

FLAGS = flags.FLAGS


def main():
    mnist = input_data.read_data_sets(FLAGS.dataset_dir, one_hot=True)

    # multiple call must add reuse=tf.AUTO_REUSE
    with tf.variable_scope('line_model', reuse=tf.AUTO_REUSE):
        x = tf.placeholder(tf.float32, [None, 784])
        # if directly use tf.Variable, system memory will increase by multiple call
        # w = tf.Variable(tf.random_normal((784, 10)))
        # b = tf.Variable(tf.random_normal((10,)))
        w = tf.get_variable('weight', [784, 10], initializer=tf.random_normal_initializer())
        b = tf.get_variable('bias', [10], initializer=tf.random_normal_initializer())
        y = tf.matmul(x, w) + b
        y_ = tf.placeholder(tf.float32, [None, 10])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.variable_scope('acc'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(10000):
            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})

            if i % 1000 == 0:
                acc, loss_ = sess.run([accuracy, loss],
                                      feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print('Step:{:5d}  validation accuracy:{:.4f}  loss:{:.1f}'.format(i, acc, loss_))
                # saver.save(sess, 'checkpoints/mnist/step-{}_acc-{}.ckpt'.format(i, acc))

        print('Finally, test acc:{:.4f}'.format(
            sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


if __name__ == '__main__':
    for loop in range(3):
        main()
