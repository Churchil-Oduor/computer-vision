#!/usr/bin/python3

import tensorflow as tf

name = tf.Variable(["churchil", "fg"], tf.string)
print(tf.rank(name))
