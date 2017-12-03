import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
import sys
import json
import time
import random
from itertools import chain
import os
import math
from model import LSTMDSSM, _START_VOCAB

random.seed(1229)

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean("read_graph", False, "Set to False to build graph.")
tf.app.flags.DEFINE_integer("symbols", 400000, "vocabulary size.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("time_log_path", 'time_log.txt', "record training time")
tf.app.flags.DEFINE_integer("neg_num", 4, "negative sample number")

FLAGS = tf.app.flags.FLAGS


def load_data(path, fname):
    print('Creating dataset...')
    data = []
    with open('%s/%s' % (path, fname)) as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            tokens = line.split()
            data.append(tokens)
    return data


def build_vocab(path, data):
    print("Creating vocabulary...")
    words = set()
    for line in data:
        for word in line:
            if len(word) == 0:
                continue
            words.add(word)
    words = list(words)
    vocab_list = _START_VOCAB + words
    FLAGS.symbols = len(vocab_list)

    print("Loading word vectors...")
    embed = np.random.normal(0.0, np.sqrt(1. / (FLAGS.embed_units)), [len(vocab_list), FLAGS.embed_units])
    # debug
    embed = np.array(embed, dtype=np.float32)
    return vocab_list, embed
    with open(os.path.join(path, 'vector.txt')) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            info = line.split()
            if info[0] not in vocab_list:
                continue
            embed[vocab_list.index(info[0])] = [float(num) for num in info[1:]]
    embed = np.array(embed, dtype=np.float32)
    return vocab_list, embed


def gen_batch_data(data):
    def padding(sent, l):
        return sent + ['_PAD'] * (l - len(sent))

    max_len = max([len(sentence) for sentence in data])
    texts, texts_length = [], []

    for item in data:
        texts.append(padding(item, max_len))
        texts_length.append(len(item))

    batched_data = {'texts': np.array(texts), 'texts_length': np.array(texts_length, dtype=np.int32)}

    return batched_data


def train(model, sess, queries, docs):
    st, ed, loss = 0, 0, .0
    lq = len(queries) / (FLAGS.neg_num + 1)
    while ed < lq:
        st, ed = ed, ed + FLAGS.batch_size if ed + FLAGS.batch_size < lq else lq
        batch_queries = gen_batch_data(queries[st:ed])
        batch_docs = gen_batch_data(docs[st*(FLAGS.neg_num + 1):ed*(FLAGS.neg_num + 1)])
        lbq = len(batch_queries['texts'])
        batch_docs['texts'] = batch_docs['texts'].reshape(lbq, FLAGS.neg_num + 1, -1)
        batch_docs['texts_length'] = batch_docs['texts_length'].reshape(lbq, FLAGS.neg_num + 1)
        outputs = model.train_step(sess, batch_queries, batch_docs)
        if math.isnan(outputs[0]) or math.isinf(outputs[0]):
            print('nan/inf detected. ')
        loss += outputs[0]
    sess.run([model.epoch_add_op])

    return loss / len(queries)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        print(FLAGS.__flags)
        data_queries = load_data(FLAGS.data_dir, 'queries.txt')
        data_docs = load_data(FLAGS.data_dir, 'docs.txt')
        vocab, embed = build_vocab(FLAGS.data_dir, data_queries + data_docs)

        model = LSTMDSSM(
            FLAGS.units,
            embed,
            FLAGS.neg_num)
        if FLAGS.log_parameters:
            model.print_parameters()

        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.word2index.insert(constant_op.constant(vocab),
                                              constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)

        summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
        total_train_time = 0.0
        while model.epoch.eval() < FLAGS.epoch:
            epoch = model.epoch.eval()
            random_idxs = range(len(data_queries))
            random.shuffle(random_idxs)
            data_queries = [data_queries[i] for i in random_idxs]
            data_docs = np.reshape(data_docs, (len(data_queries), -1))
            data_docs = [data_docs[i] for i in random_idxs]
            data_docs = np.reshape(data_docs, len(data_queries) * (FLAGS.neg_num + 1))
            start_time = time.time()
            loss = train(model, sess, data_queries, data_docs)

            epoch_time = time.time() - start_time
            total_train_time += epoch_time

            summary = tf.Summary()
            summary.value.add(tag='loss/train', simple_value=loss)
            cur_lr = model.learning_rate.eval()
            summary.value.add(tag='lr/train', simple_value=cur_lr)
            summary_writer.add_summary(summary, epoch)
            model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
            print("epoch %d learning rate %.10f epoch-time %.4f loss %.8f" % (
            epoch, cur_lr, epoch_time, loss))
        with open(os.path.join(FLAGS.train_dir, FLAGS.time_log_path), 'a') as fp:
            fp.writelines(['total training time: %f' % total_train_time])
