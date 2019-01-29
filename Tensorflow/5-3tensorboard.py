import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#每个批次的大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

#参数概要
def variable_Summaries(var):
    with tf.name_scope('Summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean) #平均值
        with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) #标准差
        tf.summary.scalar('max',tf.reduce_max(var))  #最大值
        tf.summary.scalar('min',tf.reduce_min(var)) #最小值
        tf.summary.histogram('histogram',var)  #直方图
        
#定义命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('layer'):
    #创建一个简单的神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name='w')
        variable_Summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_Summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W)+b 
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
#对数释然代价函数
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
#结果是存放在一个bool型列表中，
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #argmax 返回一维张量中最大值所在的位置
   
    with tf.name_scope('accuracy'):
         #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
            
        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))
 
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Use the retry module or similar alternatives.
WARNING:tensorflow:From <ipython-input-1-f2018b7b205c>:3: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-images-idx3-ubyte.gz
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From <ipython-input-1-f2018b7b205c>:43: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

Iter0,Testing Accuracy 0.8248
Iter1,Testing Accuracy 0.8947
Iter2,Testing Accuracy 0.9016
Iter3,Testing Accuracy 0.9058
Iter4,Testing Accuracy 0.9078
Iter5,Testing Accuracy 0.9097
Iter6,Testing Accuracy 0.9114
Iter7,Testing Accuracy 0.913
Iter8,Testing Accuracy 0.9147
Iter9,Testing Accuracy 0.9157
Iter10,Testing Accuracy 0.9179
Iter11,Testing Accuracy 0.9187
Iter12,Testing Accuracy 0.9202
Iter13,Testing Accuracy 0.919
Iter14,Testing Accuracy 0.9201
Iter15,Testing Accuracy 0.9204
Iter16,Testing Accuracy 0.9198
Iter17,Testing Accuracy 0.9206
Iter18,Testing Accuracy 0.9209
Iter19,Testing Accuracy 0.9219
Iter20,Testing Accuracy 0.9214
Iter21,Testing Accuracy 0.9217
Iter22,Testing Accuracy 0.9229
Iter23,Testing Accuracy 0.9226
Iter24,Testing Accuracy 0.9232
Iter25,Testing Accuracy 0.9234
Iter26,Testing Accuracy 0.9234
Iter27,Testing Accuracy 0.9234
Iter28,Testing Accuracy 0.9231
Iter29,Testing Accuracy 0.9236
Iter30,Testing Accuracy 0.9238
Iter31,Testing Accuracy 0.9244
Iter32,Testing Accuracy 0.9234
Iter33,Testing Accuracy 0.9239
Iter34,Testing Accuracy 0.9244
Iter35,Testing Accuracy 0.9247
Iter36,Testing Accuracy 0.9248
Iter37,Testing Accuracy 0.9255
Iter38,Testing Accuracy 0.9255
Iter39,Testing Accuracy 0.9262
Iter40,Testing Accuracy 0.9256
Iter41,Testing Accuracy 0.9276
Iter42,Testing Accuracy 0.926
Iter43,Testing Accuracy 0.9274
Iter44,Testing Accuracy 0.9264
Iter45,Testing Accuracy 0.9263
Iter46,Testing Accuracy 0.9271
Iter47,Testing Accuracy 0.9256
Iter48,Testing Accuracy 0.9271
Iter49,Testing Accuracy 0.9267
Iter50,Testing Accuracy 0.927