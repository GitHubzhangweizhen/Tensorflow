import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
#每个批次的大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#对数释然代价函数
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
#结果是存放在一个bool型列表中，
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #argmax 返回一维张量中最大值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))
 
Iter0,Testing Accuracy 0.6452
Iter1,Testing Accuracy 0.78
Iter2,Testing Accuracy 0.7979
Iter3,Testing Accuracy 0.8066
Iter4,Testing Accuracy 0.813
Iter5,Testing Accuracy 0.8165
Iter6,Testing Accuracy 0.8196
Iter7,Testing Accuracy 0.8262
Iter8,Testing Accuracy 0.8571
Iter9,Testing Accuracy 0.8706
Iter10,Testing Accuracy 0.878
Iter11,Testing Accuracy 0.8844
Iter12,Testing Accuracy 0.8876
Iter13,Testing Accuracy 0.8902
Iter14,Testing Accuracy 0.8915
Iter15,Testing Accuracy 0.8927
Iter16,Testing Accuracy 0.8938
Iter17,Testing Accuracy 0.8947
Iter18,Testing Accuracy 0.8954
Iter19,Testing Accuracy 0.8961
Iter20,Testing Accuracy 0.8968
 