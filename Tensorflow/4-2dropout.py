import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#每个批次的大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)  #设置有多少个神经元在工作 

#创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)
#中间增加隐藏层
W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000,1000],stddev=0.1))
b4 = tf.Variable(tf.zeros([1000])+0.1)
L4 = tf.nn.tanh(tf.matmul(L3_drop,W4)+b4)
L4_drop = tf.nn.dropout(L4,keep_prob)


W5 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b5 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L4_drop,W5)+b5)

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
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
            
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})

        print("Iter"+str(epoch)+",Testing Accuracy "+str(test_acc) + ",Training Accuracy "+ str(train_acc))
 
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Iter0,Testing Accuracy 0.9076,Training Accuracy0.9001091
Iter1,Testing Accuracy 0.9228,Training Accuracy0.9186
Iter2,Testing Accuracy 0.9289,Training Accuracy0.9263273
Iter3,Testing Accuracy 0.9345,Training Accuracy0.9323091
Iter4,Testing Accuracy 0.9389,Training Accuracy0.93741816
Iter5,Testing Accuracy 0.9414,Training Accuracy0.9403273
Iter6,Testing Accuracy 0.944,Training Accuracy0.9436727
Iter7,Testing Accuracy 0.9464,Training Accuracy0.9468909
Iter8,Testing Accuracy 0.9483,Training Accuracy0.9494182
Iter9,Testing Accuracy 0.9491,Training Accuracy0.9485818
Iter10,Testing Accuracy 0.9505,Training Accuracy0.9533455
Iter11,Testing Accuracy 0.9535,Training Accuracy0.9557273
Iter12,Testing Accuracy 0.9544,Training Accuracy0.9566727
Iter13,Testing Accuracy 0.9558,Training Accuracy0.95787275
Iter14,Testing Accuracy 0.9566,Training Accuracy0.96
Iter15,Testing Accuracy 0.9561,Training Accuracy0.96018183
Iter16,Testing Accuracy 0.9565,Training Accuracy0.9605273
Iter17,Testing Accuracy 0.9594,Training Accuracy0.9635636
Iter18,Testing Accuracy 0.9592,Training Accuracy0.9636
Iter19,Testing Accuracy 0.9596,Training Accuracy0.9642
Iter20,Testing Accuracy 0.9615,Training Accuracy0.9655273
Iter0,Testing Accuracy 0.9479,Training Accuracy0.9583091
Iter1,Testing Accuracy 0.9577,Training Accuracy0.97554547
Iter2,Testing Accuracy 0.9628,Training Accuracy0.9824727
Iter3,Testing Accuracy 0.9652,Training Accuracy0.9865636
Iter4,Testing Accuracy 0.9676,Training Accuracy0.9886364
Iter5,Testing Accuracy 0.9688,Training Accuracy0.9900727
Iter6,Testing Accuracy 0.9688,Training Accuracy0.99105453
Iter7,Testing Accuracy 0.9684,Training Accuracy0.99169093
Iter8,Testing Accuracy 0.9693,Training Accuracy0.9919636
Iter9,Testing Accuracy 0.9696,Training Accuracy0.9923273
Iter10,Testing Accuracy 0.97,Training Accuracy0.9925454
Iter11,Testing Accuracy 0.9701,Training Accuracy0.99285454
Iter12,Testing Accuracy 0.9705,Training Accuracy0.993
Iter13,Testing Accuracy 0.9697,Training Accuracy0.9932909
Iter14,Testing Accuracy 0.9711,Training Accuracy0.99347275
Iter15,Testing Accuracy 0.9719,Training Accuracy0.99363637
Iter16,Testing Accuracy 0.9709,Training Accuracy0.99381816
Iter17,Testing Accuracy 0.9711,Training Accuracy0.994
Iter18,Testing Accuracy 0.9711,Training Accuracy0.99414545
Iter19,Testing Accuracy 0.9711,Training Accuracy0.99436367
Iter20,Testing Accuracy 0.9713,Training Accuracy0.9945091
#3层隐藏层
Iter0,Testing Accuracy 0.9203,Training Accuracy0.91376364
Iter1,Testing Accuracy 0.9317,Training Accuracy0.92814547
Iter2,Testing Accuracy 0.9356,Training Accuracy0.93503636
Iter3,Testing Accuracy 0.9431,Training Accuracy0.94214547
Iter4,Testing Accuracy 0.9456,Training Accuracy0.9452
Iter5,Testing Accuracy 0.9478,Training Accuracy0.9485091
Iter6,Testing Accuracy 0.9494,Training Accuracy0.95092726
Iter7,Testing Accuracy 0.9539,Training Accuracy0.95354545
Iter8,Testing Accuracy 0.9521,Training Accuracy0.9552909
Iter9,Testing Accuracy 0.9557,Training Accuracy0.9587455
Iter10,Testing Accuracy 0.956,Training Accuracy0.9593818
Iter11,Testing Accuracy 0.9572,Training Accuracy0.9614
Iter12,Testing Accuracy 0.9575,Training Accuracy0.9626909
Iter13,Testing Accuracy 0.9602,Training Accuracy0.9646364
Iter14,Testing Accuracy 0.9604,Training Accuracy0.9656364
Iter15,Testing Accuracy 0.9624,Training Accuracy0.96687275
Iter16,Testing Accuracy 0.9631,Training Accuracy0.9678364
Iter17,Testing Accuracy 0.9652,Training Accuracy0.9689818
Iter18,Testing Accuracy 0.9656,Training Accuracy0.96958184
Iter19,Testing Accuracy 0.966,Training Accuracy0.97116363
Iter20,Testing Accuracy 0.9671,Training Accuracy0.9715818
  File "<ipython-input-5-d7a259369b4f>", line 1
    Iter0,Testing Accuracy 0.9479,Training Accuracy0.9583091
                         ^
SyntaxError: invalid syntax