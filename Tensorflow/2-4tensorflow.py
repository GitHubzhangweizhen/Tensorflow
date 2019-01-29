import tensorflow as tf
import numpy as np
#使用numpy生成100个随机点，样本
x_data = np.random.rand(100)
y_data =x_data*0.3 + 0.2

#构造一个线性模型,这个线性模型的截距和斜率都是变量，需要优化这两个变量，使得接近或者等于上面的样本点
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

#二次代价函数,误差的平方再求平均值
loss = tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)
#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(301):
        sess.run(train)
        if step%20 ==0:
            print(step,sess.run([k,b]))
0 [0.087366275, 0.14406608]
20 [0.2207689, 0.24532421]
40 [0.25387508, 0.22638583]
60 [0.27314806, 0.21536069]
80 [0.28436795, 0.20894234]
100 [0.2908997, 0.20520584]
120 [0.2947021, 0.20303066]
140 [0.2969158, 0.20176433]
160 [0.2982045, 0.20102711]
180 [0.29895476, 0.20059794]
200 [0.2993915, 0.2003481]
220 [0.29964575, 0.20020264]
240 [0.29979375, 0.20011799]
260 [0.29987994, 0.20006868]
280 [0.29993019, 0.20003994]
300 [0.29995936, 0.20002323]
 