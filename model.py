import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from cells import SimpleLSTMCell

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class LSTMDSSM(object):
    """
    The LSTM-DSSM model refering to the paper: Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.
    papaer available at: https://arxiv.org/abs/1502.06922
    """

    def __init__(self,
                 num_lstm_units,
                 embed,
                 neg_num=4,
                 gradient_clip_threshold=5.0):
        self.queries = tf.placeholder(dtype=tf.string, shape=[None, None])  # shape: batch*len
        self.queries_length = tf.placeholder(dtype=tf.int32, shape=[None])  # shape: batch
        self._docs = tf.placeholder(dtype=tf.string, shape=[None, neg_num + 1, None])  # shape: batch*(neg_num + 1)*len
        self._docs_length = tf.placeholder(dtype=tf.int32, shape=[None, neg_num + 1])  # shape: batch*(neg_num + 1)
        self.docs = tf.transpose(self._docs, [1, 0, 2])  # shape: (neg_num + 1)*batch*len
        self.docs_length = tf.transpose(self._docs_length)  # shape: batch*(neg_num + 1)

        self.word2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True
        )

        self.learning_rate = tf.Variable(0.001, trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.momentum = tf.Variable(0.9, trainable=False, dtype=tf.float32)

        self.index_queries = self.word2index.lookup(self.queries)  # batch*len
        self.index_docs = [self.word2index.lookup(doc) for doc in tf.unstack(self.docs)]

        self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        self.embed_queries = tf.nn.embedding_lookup(self.embed, self.index_queries)
        self.embed_docs = [tf.nn.embedding_lookup(self.embed, index_doc) for index_doc in self.index_docs]

        with tf.variable_scope('query_lstm'):
            self.cell_q = SimpleLSTMCell(num_lstm_units)
        with tf.variable_scope('doc_lstm'):
            self.cell_d = SimpleLSTMCell(num_lstm_units)

        self.states_q = dynamic_rnn(self.cell_q, self.embed_queries, self.queries_length, dtype=tf.float32,
                                         scope="simple_lstm_cell_query")[1][1]  # shape: batch*num_units
        self.states_d = [dynamic_rnn(self.cell_d, self.embed_docs[i], self.docs_length[i], dtype=tf.float32,
                                            scope="simple_lstm_cell_doc")[1][1] for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch*num_units
        self.queries_norm = tf.reduce_sum(self.states_q, axis=1)
        self.docs_norm = [tf.reduce_sum(self.states_d[i], axis=1) for i in range(neg_num + 1)]
        self.prods = [tf.reduce_sum(tf.multiply(self.states_q, self.states_d[i]), axis=1) for i in range(neg_num + 1)]
        self.sims = [(self.prods[i] / (self.queries_norm * self.docs_norm[i])) for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch
        self.sims = tf.transpose(tf.convert_to_tensor(self.sims))  # shape: batch*(neg_num + 1)
        self.gamma = tf.Variable(initial_value=1.0, expected_shape=[], dtype=tf.float32)  # scaling factor according to the paper
        self.sims = self.sims * self.gamma
        self.prob = tf.nn.softmax(self.sims)
        self.hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])
        self.loss = -tf.reduce_mean(tf.log(self.hit_prob))

        self.params = tf.trainable_variables()
        opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)  # use Nesterov's method, according to the paper
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, gradient_clip_threshold)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def train_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self._docs: docs['texts'],
                      self._docs_length: docs['texts_length']}

        output_feed = [self.loss, self.update, self.states_q, self.states_d, self.queries_norm, self.docs_norm, self.prods, self.sims, self.gamma, self.prob, self.hit_prob]
        return session.run(output_feed, input_feed)
