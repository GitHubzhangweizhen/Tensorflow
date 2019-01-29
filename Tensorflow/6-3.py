import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


#每个批次的大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial)
#初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#改变x的格式转为4D的向量
x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一个卷积层的权值和偏置值
W_conv1 = weight_variable([5,5,1,32]) #5*5的采样窗口，32个卷积核从一个平面抽取特征
b_conv1 = bias_variable([32])  #每一个卷积核一个偏置值

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #进行max-pooling

#初始化第二个卷积层的权值和偏置值
W_conv2 = weight_variable([5,5,32,64]) #5*5的采样窗口，32个卷积核从一个平面抽取特征
b_conv2 = bias_variable([64])  #每一个卷积核一个偏置值

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #进行max-pooling

#28*28的图片第一次卷积之后还是28*28，第一次池化后变成14*14，第二次卷积后还是14*14，第二次池化之后变成7*7
#通过上面的操作之后得到64张7*7的平面

#初始化第一个全连接层的权值和偏置值
W_fc1 = weight_variable([7*7*64,1024])  #上一层有7*7*64个神经元，全连接有1024个神经元
b_fc1 = bias_variable([1024])

#把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)  #设置有多少个神经元在工作 
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#定义第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#对数释然代价函数
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
#使用优化器梯度下降法
#train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)  #这里学习率是0.01
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
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
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})           
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(test_acc))
WARNING:tensorflow:From C:\Program Files\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Use the retry module or similar alternatives.
WARNING:tensorflow:From <ipython-input-1-cf9192c9163b>:3: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From C:\Program Files\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From C:\Program Files\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-images-idx3-ubyte.gz
WARNING:tensorflow:From C:\Program Files\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
WARNING:tensorflow:From C:\Program Files\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From C:\Program Files\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From <ipython-input-1-cf9192c9163b>:70: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

Iter0,Testing Accuracy 0.8628
Iter1,Testing Accuracy 0.9639
Iter2,Testing Accuracy 0.9786
Iter3,Testing Accuracy 0.9801
Iter4,Testing Accuracy 0.985
Iter5,Testing Accuracy 0.986
Iter6,Testing Accuracy 0.9842
Iter7,Testing Accuracy 0.988
Iter8,Testing Accuracy 0.9876
Iter9,Testing Accuracy 0.9881
Iter10,Testing Accuracy 0.9895
Iter11,Testing Accuracy 0.9894
Iter12,Testing Accuracy 0.9894
Iter13,Testing Accuracy 0.9898
Iter14,Testing Accuracy 0.991
Iter15,Testing Accuracy 0.9914
Iter16,Testing Accuracy 0.9912
Iter17,Testing Accuracy 0.9907
Iter18,Testing Accuracy 0.9924
Iter19,Testing Accuracy 0.9919
Iter20,Testing Accuracy 0.9914