import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#运行次数
max_steps =1001
#图片数量
image_num =3000

DIR="C:/Users/Administrator/Documents/"
sess = tf.Session()
#载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='emdedding')
#每个批次的大小
# batch_size = 100
# n_batch = mnist.train.num_examples // batch_size

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
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)
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
sess.run(init)
#结果是存放在一个bool型列表中，
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #argmax 返回一维张量中最大值所在的位置
   
    with tf.name_scope('accuracy'):
         #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
#生成metadata文件

if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')

with open(DIR + 'projector/projector/metadata.tsv','w') as f:
 
    labels = sess.run(tf.argmax(mnist.test.labels[:],1))
    
    for i in range(image_num):
        f.write(str(labels[i])+'\n')
#合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector',sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

for i in range(max_steps):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,'step%03d' % i)
    projector_writer.add_summary(summary,1)
  
    if i%100==0:
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(i)+",Testing Accuracy "+str(acc))
saver.save(sess,DIR + 'projector/projector/a_model.ckpt',global_step=max_steps)
projector_writer.close()
sess.close()

# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter('logs/',sess.graph)
#     for epoch in range(51):
#         for batch in range(n_batch):
#             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#             summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
            
#         writer.add_summary(summary,epoch)
#         acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
#         print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))
 
WARNING:tensorflow:From G:\Anacon\bb\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Use the retry module or similar alternatives.
WARNING:tensorflow:From <ipython-input-1-f947d25aa6f5>:4: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
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
WARNING:tensorflow:From <ipython-input-1-f947d25aa6f5>:54: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

Iter0,Testing Accuracy 0.1889
Iter100,Testing Accuracy 0.7661
Iter200,Testing Accuracy 0.8032
Iter300,Testing Accuracy 0.8124
Iter400,Testing Accuracy 0.8201
Iter500,Testing Accuracy 0.8232
Iter600,Testing Accuracy 0.8263
Iter700,Testing Accuracy 0.8283
Iter800,Testing Accuracy 0.8417
Iter900,Testing Accuracy 0.8674
Iter1000,Testing Accuracy 0.8789
 