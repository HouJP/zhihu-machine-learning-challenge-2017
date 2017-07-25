#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 7/25/17 12:41 AM
# @Author  : Jianpeng Hou
# @Email   : houjp1992@gmail.com


"""
Change window size of Conv: [1-10, 11, 13, 15, 20]
"""


from keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, model_from_json
from keras import optimizers
import tensorflow as tf
from keras import backend as K

from bin.utils import LogUtil
from bin.text_cnn.loss import binary_crossentropy_sum
from bin.text_cnn.data_helpers import load_embedding


def init_text_cnn(config):
    # set number of cores
    mode = config.get('ENVIRONMENT', 'mode')
    LogUtil.log('INFO', 'mode=%s' % mode)
    if 'cpu' == mode:
        num_cores = config.getint('ENVIRONMENT', 'num_cores')
        tf_config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores,
                                   allow_soft_placement=True, device_count={'CPU': num_cores})
        session = tf.Session(config=tf_config)
        K.set_session(session)
    elif 'gpu' == mode:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        K.set_session(sess)

    # load word embedding file
    word_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'word_embedding_fn'))
    word_embedding_index, word_embedding_matrix = load_embedding(word_embedding_fp)
    # load char embedding file
    char_embedding_fp = '%s/%s' % (config.get('DIRECTORY', 'embedding_pt'),
                                   config.get('TITLE_CONTENT_CNN', 'char_embedding_fn'))
    char_embedding_index, char_embedding_matrix = load_embedding(char_embedding_fp)
    # init model
    title_word_length = config.getint('TITLE_CONTENT_CNN', 'title_word_length')
    content_word_length = config.getint('TITLE_CONTENT_CNN', 'content_word_length')
    title_char_length = config.getint('TITLE_CONTENT_CNN', 'title_char_length')
    content_char_length = config.getint('TITLE_CONTENT_CNN', 'content_char_length')
    fs_btm_tw_cw_length = config.getint('TITLE_CONTENT_CNN', 'fs_btm_tw_cw_length')
    fs_btm_tc_length = config.getint('TITLE_CONTENT_CNN', 'fs_btm_tc_length')
    class_num = config.getint('TITLE_CONTENT_CNN', 'class_num')
    optimizer_name = config.get('TITLE_CONTENT_CNN', 'optimizer_name')
    lr = float(config.get('TITLE_CONTENT_CNN', 'lr'))
    metrics = config.get('TITLE_CONTENT_CNN', 'metrics').split()
    model = TitleContentCNN(title_word_length=title_word_length,
                            content_word_length=content_word_length,
                            title_char_length=title_char_length,
                            content_char_length=content_char_length,
                            fs_btm_tw_cw_length=fs_btm_tw_cw_length,
                            fs_btm_tc_length=fs_btm_tc_length,
                            class_num=class_num,
                            word_embedding_matrix=word_embedding_matrix,
                            char_embedding_matrix=char_embedding_matrix,
                            optimizer_name=optimizer_name,
                            lr=lr,
                            metrics=metrics)

    return model, word_embedding_index, char_embedding_index


class TitleContentCNN(object):
    def __init__(self,
                 title_word_length,
                 content_word_length,
                 title_char_length,
                 content_char_length,
                 fs_btm_tw_cw_length,
                 fs_btm_tc_length,
                 class_num,
                 word_embedding_matrix,
                 char_embedding_matrix,
                 optimizer_name,
                 lr,
                 metrics):
        # set attributes
        self.title_word_length = title_word_length
        self.content_word_length = content_word_length
        self.title_char_length = title_char_length
        self.content_char_length = content_char_length
        self.fs_btm_tw_cw_length = fs_btm_tw_cw_length
        self.fs_btm_tc_length = fs_btm_tc_length
        self.class_num = class_num
        self.word_embedding_matrix = word_embedding_matrix
        self.char_embedding_matrix = char_embedding_matrix
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.metrics = metrics
        # Placeholder for input (title and content)
        title_word_input = Input(shape=(title_word_length,), dtype='int32', name="title_word_input")
        cont_word_input = Input(shape=(content_word_length,), dtype='int32', name="content_word_input")

        title_char_input = Input(shape=(title_char_length,), dtype='int32', name="title_char_input")
        cont_char_input = Input(shape=(content_char_length,), dtype='int32', name="content_char_input")

        # Embedding layer
        word_embedding_layer = Embedding(len(word_embedding_matrix),
                                         256,
                                         weights=[word_embedding_matrix],
                                         trainable=True, name='word_embedding')
        title_word_emb = word_embedding_layer(title_word_input)
        cont_word_emb = word_embedding_layer(cont_word_input)

        char_embedding_layer = Embedding(len(char_embedding_matrix),
                                         256,
                                         weights=[char_embedding_matrix],
                                         trainable=True, name='char_embedding')
        title_char_emb = char_embedding_layer(title_char_input)
        cont_char_emb = char_embedding_layer(cont_char_input)

        # Create a convolution + max pooling layer
        title_content_features = list()
        for win_size in range(1, 8):
            # batch_size x doc_len x embed_size
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(100, win_size, activation='relu', padding='same')(title_word_emb)))
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(100, win_size, activation='relu', padding='same')(cont_word_emb)))
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(100, win_size, activation='relu', padding='same')(title_char_emb)))
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(100, win_size, activation='relu', padding='same')(cont_char_emb)))

        # add btm_tw_cw features + btm_tc features
        fs_btm_tw_cw_input = Input(shape=(fs_btm_tw_cw_length,), dtype='float32', name="fs_btm_tw_cw_input")
        fs_btm_tc_input = Input(shape=(fs_btm_tc_length,), dtype='float32', name="fs_btm_tc_input")
        fs_btm_raw_features = concatenate([fs_btm_tw_cw_input, fs_btm_tc_input])
        fs_btm_emb_features = Dense(1024, activation='relu', name='fs_btm_embedding')(fs_btm_raw_features)
        fs_btm_emb_features = Dropout(0.5, name='fs_btm_embedding_dropout')(fs_btm_emb_features)
        title_content_features.append(fs_btm_emb_features)

        title_content_features = concatenate(title_content_features)

        # Full connection
        title_content_features = Dense(3600, activation='relu', name='fs_embedding')(title_content_features)
        title_content_features = Dropout(0.5, name='fs_embedding_dropout')(title_content_features)

        # Prediction
        preds = Dense(class_num, activation='sigmoid', name='prediction')(title_content_features)

        self._model = Model([title_word_input,
                             cont_word_input,
                             title_char_input,
                             cont_char_input,
                             fs_btm_tw_cw_input,
                             fs_btm_tc_input], preds)
        if 'rmsprop' == optimizer_name:
            optimizer = optimizers.RMSprop(lr=lr)
        elif 'adam' == optimizer_name:
            optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        else:
            optimizer = None
        self._model.compile(loss=binary_crossentropy_sum, optimizer=optimizer, metrics=metrics)
        self._model.summary()

    def save(self, model_fp):
        model_json = self._model.to_json()
        with open('%s.json' % model_fp, 'w') as json_file:
            json_file.write(model_json)
        self._model.save_weights('%s.h5' % model_fp)
        LogUtil.log('INFO', 'save model (%s) to disk done' % model_fp)

    def load(self, model_fp):
        # load json and create model
        json_file = open('%s.json' % model_fp, 'r')
        model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(model_json)
        # load weights into new model
        self._model.load_weights('%s.h5' % model_fp)
        LogUtil.log('INFO', 'load model (%s) from disk done' % model_fp)

    def fit(self, x, y, batch_size, epochs=1, validation_data=None):
        self._model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, x, batch_size, verbose):
        return self._model.predict(x, batch_size=batch_size, verbose=verbose)
