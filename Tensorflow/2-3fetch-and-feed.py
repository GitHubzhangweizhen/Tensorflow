import tensorflow as tf
#Fetch 可以同时执行多个op
input1 =tf.constant(3.0)
input2 =tf.constant(2.0)
input3 =tf.constant(5.0)

add =tf.add(input2,input3)
mul =tf.multiply(input1,add)

with tf.Session() as sess:
    result =sess.run([mul,add])
    print(result)
[21.0, 7.0]
#Feed 创建占位符
input1 =tf.placeholder(tf.float32)
input2 =tf.placeholder(tf.float32)
output =tf.multiply(input1,input2)
with tf.Session() as sess:
    #Feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[8.0],input2:[2.0]}))
[16.]