import tensorflow as tf
x =tf.Variable([1,2])
a =tf.constant([3,3])
#增加一个减法op
sub =tf.subtract(x,a)
#增加一个加法
add =tf.add(x,sub)
#初始化变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
[-2 -1]
[-1  1]
#创建一个变量，初始化为0
state =tf.Variable(0,name='counter')
#创建一个op,作用是使state+1
new_value =tf.add(state,1)
update =tf.assign(state,new_value)
#初始化变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for i in range(5):
        sess.run(update)
        print(sess.run(state))
0
1
2
3
4
5
 