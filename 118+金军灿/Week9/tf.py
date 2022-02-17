import tensorflow as tf

# 查看tensorflow的版本
print(tf.__version__)

# 创建一个常量operation，
matrix1 = tf.constant([[3.0, 3.0]])
matrix2 = tf.constant([[2.0], [2.0]])

# 执行矩阵乘法
product = tf.matmul(matrix1, matrix2)

"""
    tensorflow默认构建图，操作算子op作为图的结点
    上述构建了两个常量算子和矩阵乘算子
    为了进行矩阵相乘得到结果，必须启动计算图
    启动图首先需要创建一个Session会话，如果没有传入参数，会话构造器默认启动图
"""
sess = tf.Session()
# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print(result)
sess.close()

'''
session对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 “with” 代码块来自动完成关闭动作.

with tf.Session() as sess:
  result = sess.run([product])
  print (result)
'''

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.0]}))


input1 = tf.constant([2.0])
input2 = tf.constant([3.0])
input3 = tf.constant([5.0])
intermedian = tf.add(input1,input2)
mul = tf.multiply(intermedian,input3)

with tf.Session() as sess:
    result = sess.run([intermedian,mul])
    print(result)
