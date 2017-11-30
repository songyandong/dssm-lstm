import tensorflow as tf


class SimpleLSTMCell(tf.contrib.rnn.RNNCell):
    """
    The simpler version of LSTM cell with forget gate set to 1, according to the paper.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "simple_lstm_cell", reuse=self._reuse):
            c, h = state
            if not hasattr(self, '_wi'):
                self._wi = tf.get_variable('simple_lstm_cell_wi', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bi = tf.get_variable('simple_lstm_cell_bi', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(0.0))
                self._wo = tf.get_variable('simple_lstm_cell_wo', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bo = tf.get_variable('simple_lstm_cell_bo', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(0.0))
                self._wc = tf.get_variable('simple_lstm_cell_wc', dtype=tf.float32, shape=[inputs.get_shape()[-1] + h.get_shape()[-1], self._num_units], initializer=tf.orthogonal_initializer())
                self._bc = tf.get_variable('simple_lstm_cell_bc', dtype=tf.float32, shape=[self._num_units], initializer=tf.constant_initializer(0.0))
            i = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wi) + self._bi)
            o = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, h], 1), self._wo) + self._bo)
            _c = self._activation(tf.matmul(tf.concat([inputs, h], 1), self._wc) + self._bc)
            # remove forget gate according to the paper
            new_c = c + i * _c
            new_h = o * self._activation(new_c)

            return new_h, (new_c, new_h)