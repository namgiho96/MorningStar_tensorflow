import tensorflow.compat.v1 as tf
import numpy as np
from pandas.io.parsers import read_csv

tf.disable_v2_behavior()

# 1.모델을 만든다
# 2.초기화를 한다


"""
       avgTemp = X
       minTemp = X
       maxTemp = X
       rainFall = X
       avgPrice = Y  
"""


class Cabbage:
    def model(self):
        tf.global_variables_initializer()  # 데이터를 초기화 한다.
        data = read_csv('cabbage_price.csv', sep=',')  # 데이터를 가지고 온다
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:, 1:-1]
        y_data = xy[:, [-1]]  # all 전체 다 가져온다 [-1]
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b  # 뉴런생성 하는 식
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost=오차
        optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)  # tensorFlow 2.0v 때는 Adam을 쓴다
        train = optmizer.minimize(cost)
        with tf.Session() as sess:  # 이부분이 런닝 과정이다.
            sess.run(tf.global_variables_initializer())  # 교욱을 시킬때 마다 초기화를 시킨다
            for step in range(100000):
                cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
                if step % 500 == 0:
                    print(f' step: {step}, cost: {cost_} ')  # cost가 줄어들수록
                    print(f' price : {hypothesis}')  # 가상가격을 찍어준다
            saver = tf.train.Saver()
            saver.save(sess, 'cabbage.ckpt')

    def initialize(self, avgTemp, minTemp, maxTemp, rainFall):
        self.avgTemp = avgTemp
        self.minTemp = minTemp
        self.maxTemp = maxTemp
        self.rainFall = rainFall

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'cabbage/cabbage.ckpt')
            data = [[self.avgTemp, self.minTemp, self.maxTemp, self.rainFall], ]  # 매트릭스 구조 [[]] Tensor
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
        return int(dict[0])


if __name__ == '__main__':
    cabbage = Cabbage()
    # cabbage.model()
    # cabbage.service()
