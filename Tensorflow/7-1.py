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
#命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])
    with tf.name_scope('x_image'):
        #改变x的格式转为4D的向量
        x_image = tf.reshape(x,[-1,28,28,1],name='x_image')

with tf.name_scope('Conv1'):
    #初始化第一个卷积层的权值和偏置值
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5,5,1,32],name='W_conv1') #5*5的采样窗口，32个卷积核从一个平面抽取特征
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32],name='b_conv1')  #每一个卷积核一个偏置值
    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1) #进行max-pooling
with tf.name_scope('Conv2'):
    #初始化第二个卷积层的权值和偏置值
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5,5,32,64],name='W_conv2') #5*5的采样窗口，32个卷积核从一个平面抽取特征
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64],name='b_conv2')  #每一个卷积核一个偏置值
    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2) #进行max-pooling
#28*28的图片第一次卷积之后还是28*28，第一次池化后变成14*14，第二次卷积后还是14*14，第二次池化之后变成7*7
#通过上面的操作之后得到64张7*7的平面
with tf.name_scope('fc1'):
    #初始化第一个全连接层的权值和偏置值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64,1024],name='W_fc1')  #上一层有7*7*64个神经元，全连接有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024],name='b_fc1')
    #把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64],name='h_pool2_flat')
    #求第一个全连接层的输出
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')  #设置有多少个神经元在工作 
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob,name='h_fc1_drop')

with tf.name_scope('fc2'):
    #定义第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024,10],name='W_fc2')
    with tf.name_scope('b_fc2'): 
        b_fc2 = bias_variable([10],name='b_fc2')
    with tf.name_scope('softmax'):
        #输出
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='loss')

#对数释然代价函数
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
#使用优化器梯度下降法
#train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)  #这里学习率是0.01
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()
#结果是存放在一个bool型列表中，
with tf.name_scope('accuracy'):
    
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #argmax 返回一维张量中最大值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    for i in range(1001):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})      
            #记录测试训练集计算的参数
            summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys.keep_prob:1.0})
            train_writer.add_summary(summary,i)
            #记录测试训练集计算的参数
            batch_xs,batch_ys = mnist.test.next_batch(batch_size)    
          
            summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys.keep_prob:1.0})
            test_writer.add_summary(summary,i)
            if i%100==0:
                test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
                train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.test.labels,keep_prob:1.0})
                print("Iter"+str(i)+",Testing Accuracy "+str(test_acc)+ ",Training Accuracy=" +str(train_acc))