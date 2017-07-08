#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 10:31
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.core import Permute, Flatten, Lambda
from keras.models import Model, model_from_json
from keras import backend as K

from bin.utils import LogUtil
from loss import binary_crossentropy_sum


class TitleContentCNN(object):
    def __init__(self,
                 title_word_length,
                 content_word_length,
                 title_char_length,
                 content_char_length,
                 title_word_topk,
                 content_word_topk,
                 title_char_topk,
                 content_char_topk,
                 class_num,
                 filter_num,
                 word_embedding_matrix,
                 char_embedding_matrix,
                 optimizer,
                 metrics):
        # set attributes
        self.title_word_length = title_word_length
        self.content_word_length = content_word_length
        self.title_char_length = title_char_length
        self.content_char_length = content_char_length
        self.title_word_topk = title_word_topk
        self.content_word_topk = content_word_topk
        self.title_char_topk = title_char_topk
        self.content_char_topk = content_char_topk
        self.class_num = class_num
        self.filter_num = filter_num
        self.word_embedding_matrix = word_embedding_matrix
        self.char_embedding_matrix = char_embedding_matrix
        self.optimizer = optimizer
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
        for win_size in range(2, 6):
            # batch_size x doc_len x nb_filter
            title_word_conv = Conv1D(filter_num, win_size, activation='relu', padding='same')(title_word_emb)
            # batch_size x nb_filter x doc_len
            title_word_conv_swap = Permute((2, 1))(title_word_conv)
            # batch_size x nb_filter x topk
            title_word_topk_out = Lambda(lambda y: K.tf.nn.top_k(y, k=title_word_topk)[0],
                                         output_shape=(filter_num, title_word_topk,))(title_word_conv_swap)
            # batch_size x (nb_filter x topk)
            title_word_flt = Flatten()(title_word_topk_out)
            title_content_features.append(title_word_flt)

            # batch_size x doc_len x nb_filter
            cont_word_conv = Conv1D(filter_num, win_size, activation='relu', padding='same')(cont_word_emb)
            # batch_size x nb_filter x doc_len
            cont_word_conv_swap = Permute((2, 1))(cont_word_conv)
            # batch_size x nb_filter x topk
            cont_word_topk_out = Lambda(lambda y: K.tf.nn.top_k(y, k=content_word_topk)[0],
                                        output_shape=(filter_num, content_word_topk,))(cont_word_conv_swap)
            # batch_size x (nb_filter x topk)
            cont_word_flt = Flatten()(cont_word_topk_out)
            title_content_features.append(cont_word_flt)

            # batch_size x doc_len x nb_filter
            title_char_conv = Conv1D(filter_num, win_size, activation='relu', padding='same')(title_char_emb)
            # batch_size x nb_filter x doc_len
            title_char_conv_swap = Permute((2, 1))(title_char_conv)
            # batch_size x nb_filter x topk
            title_char_topk_out = Lambda(lambda y: K.tf.nn.top_k(y, k=title_char_topk)[0],
                                         output_shape=(filter_num, title_char_topk,))(title_char_conv_swap)
            # batch_size x (nb_filter x topk)
            title_char_flt = Flatten()(title_char_topk_out)
            title_content_features.append(title_char_flt)

            # batch_size x doc_len x nb_filter
            cont_char_conv = Conv1D(filter_num, win_size, activation='relu', padding='same')(cont_char_emb)
            # batch_size x nb_filter x doc_len
            cont_char_conv_swap = Permute((2, 1))(cont_char_conv)
            # batch_size x nb_filter x topk
            cont_char_topk_out = Lambda(lambda y: K.tf.nn.top_k(y, k=content_char_topk)[0],
                                        output_shape=(filter_num, content_char_topk,))(cont_char_conv_swap)
            # batch_size x (nb_filter x topk)
            cont_char_flt = Flatten()(cont_char_topk_out)
            title_content_features.append(cont_char_flt)
        title_content_features = concatenate(title_content_features)

        # Full connection
        title_cont_features = Dense(1024, activation='relu')(title_content_features)

        # Prediction
        preds = Dense(class_num, activation='sigmoid')(title_cont_features)

        self._model = Model([title_word_input, cont_word_input, title_char_input, cont_char_input], preds)
        self._model.compile(loss=binary_crossentropy_sum, optimizer=optimizer, metrics=metrics)
        # self._model.summary()

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
        # compile model
        self._model.compile(loss=binary_crossentropy_sum, optimizer=self.optimizer, metrics=self.metrics)
        LogUtil.log('INFO', 'load model (%s) from disk done' % model_fp)

    def fit(self, x, y, batch_size=32, epochs=1, validation_data=None):
        self._model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, x, batch_size, verbose):
        return self._model.predict(x, batch_size=batch_size, verbose=verbose)
