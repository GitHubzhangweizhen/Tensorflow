import tensorflow as tf
matrix1 = tf.constant([[3, 3]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2],[2]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
[[12]]
 