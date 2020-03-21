import tensorflow.compat.v1 as tf
import numpy as np
from pandas.io.parsers import read_csv

"""
Weight (kilograms)
Age (Years)

Blood fat content
"""


class Blood:
    def initialize(self, weigth, age):
        self._weigth = weigth
        self._age = age

    @staticmethod
    def raw_data():
        tf.set_random_seed(777)
        return np.genfromtxt('blood.txt', skip_header=36)

    @staticmethod
    def model(raw_data):
        tf.global_variables_initializer()  # 데이터를 초기화 한다.
        x_data = np.array(raw_data[:, 2:4], dtype=np.float32)  # 전처리과정
        y_data = np.array(raw_data[:, 4], dtype=np.float32)  # 전처리과정
        y_data = y_data.reshape(25, 1)  # 전처리과정
        X = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
        Y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b  # 뉴런생성 하는 식
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost=오차
        optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)  # tensorFlow 2.0v 때는 Adam을 쓴다
        train = optmizer.minimize(cost)
        sess = tf.Session()  # 이부분이 런닝 과정이다.
        sess.run(tf.global_variables_initializer())  # 교욱을 시킬때 마다 초기화를 시킨다
        cost_history = []
        for step in range(2000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
            if step % 500 == 0:
                cost_history.append(sess.run(cost, {X: x_data, Y: y_data}))
        saver = tf.train.Saver()
        saver.save(sess, 'blood.ckpt')

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 2])
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'blood/blood.ckpt')
            val = sess.run(tf.matmul(X, W) + b, {X: [[self._weigth, self._age], ]})

        print('혈중 지방농도: {}'.format(val))
        if val < 150:
            result = '정상'
        elif 150 <= val < 200:
            result = '경계역 중성지방혈증'
        elif 200 <= val < 500:
            result = '고 중성지방혈증'
        elif 500 <= val < 1000:
            result = '초고 중성지방혈증'
        elif 1000 <= val:
            result = '췌장염 발병 가능성 고도화'
        print(result)
        return result

# if __name__ == '__main__':
#     blood = Blood()
    # raw_data = blood.raw_data()
    # blood.model(raw_data)
    # blood.initialize(100, 30)
    # blood.service()
