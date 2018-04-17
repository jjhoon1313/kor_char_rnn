# -*- coding: utf-8 -*-

import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):

    """

    :param inputs: Attention inputs
                    In case of RNN, this must be RNN outputs 'Tensor'
                    if time_major == False :
                         [batch_size, max_time, cell.output_size]
                    else :
                         [max_time, batch_size, cell.output_size]

    :param attention_size: Linear size of attention weights.

    :param time_major: shape format of the 'inputs' Tensors.

    :param return_alphas: for visualization purpose.

    :return:
            In case of RNN, this will be a 'Tensor' shaped:
            [batch_size, cell.output_size]
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:

        inputs = tf.array_ops.transpose(inputs,[1,0,2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # One fully connected layer with non-linear activation for each of the hidden states;
    # The shape of 'v' is (B*T,A), where A=attention_size
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.expand_dims(b_omega, 0))

    # For each of the B*T hidden states its vector of size A from 'v' is reduced with 'u' vector
    vu = tf.matmul(v, tf.expand_dims(u_omega, -1))  # (B*T,1) shape
    vu = tf.reshape(vu, tf.shape(inputs)[:2])       # (B*T) shape
    alphas = tf.nn.softmax(vu, name='alphas')                      # (B*T) shape also

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1, name='output')


    if not return_alphas:
        return output
    else:
        return output, alphas