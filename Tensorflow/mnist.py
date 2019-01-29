import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
#每个批次的大小
batch_size = 2
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
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
    for epoch in range(40):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))
 
Iter0,Testing Accuracy 0.9167
Iter1,Testing Accuracy 0.9186
Iter2,Testing Accuracy 0.9225
Iter3,Testing Accuracy 0.9269
Iter4,Testing Accuracy 0.9277
Iter5,Testing Accuracy 0.9263
Iter6,Testing Accuracy 0.9279
Iter7,Testing Accuracy 0.9272
Iter8,Testing Accuracy 0.9281
Iter9,Testing Accuracy 0.9297
Iter10,Testing Accuracy 0.9284
Iter11,Testing Accuracy 0.9277
Iter12,Testing Accuracy 0.9294
Iter13,Testing Accuracy 0.9282
Iter14,Testing Accuracy 0.9263
Iter15,Testing Accuracy 0.9278
Iter16,Testing Accuracy 0.9255
Iter17,Testing Accuracy 0.9299
Iter18,Testing Accuracy 0.9306
Iter19,Testing Accuracy 0.9282
Iter20,Testing Accuracy 0.9298
Iter21,Testing Accuracy 0.9296
Iter22,Testing Accuracy 0.9305
Iter23,Testing Accuracy 0.9287
Iter24,Testing Accuracy 0.929
Iter25,Testing Accuracy 0.9298
Iter26,Testing Accuracy 0.9295
Iter27,Testing Accuracy 0.9297
Iter28,Testing Accuracy 0.9311
Iter29,Testing Accuracy 0.9294
Iter30,Testing Accuracy 0.9306
Iter31,Testing Accuracy 0.9307
Iter32,Testing Accuracy 0.9289
Iter33,Testing Accuracy 0.9312
Iter34,Testing Accuracy 0.929
Iter35,Testing Accuracy 0.9273
Iter36,Testing Accuracy 0.9302
Iter37,Testing Accuracy 0.9309
Iter38,Testing Accuracy 0.9295
Iter39,Testing Accuracy 0.932
 