#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 10:31
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from keras.layers import Dense, Input, Embedding, Conv1D, merge, GlobalMaxPooling1D
from keras.models import Model, model_from_json
from loss import binary_crossentropy_sum
from utils import LogUtil


class TitleContentCNN(object):

    def __init__(self,
                 title_length,
                 content_length,
                 class_num,
                 embedding_matrix,
                 optimizer,
                 metrics):
        # set attributes
        self.title_length = title_length
        self.content_length = content_length
        self.class_num = class_num
        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer
        self.metrics = metrics
        # Placeholder for input (title and content)
        title_input = Input(shape=(title_length, ), dtype='int32', name="title_word_input")
        cont_input = Input(shape=(content_length, ), dtype='int32', name="content_word_input")

        # Embedding layer
        embedding_layer = Embedding(len(embedding_matrix), 256, weights=[embedding_matrix], trainable=True)
        title_emb = embedding_layer(title_input)
        cont_emb = embedding_layer(cont_input)

        # Create a convolution + max pooling layer
        title_cont_conv = list()
        for win_size in range(2, 6):
            title_cont_conv.append(Conv1D(128, win_size, activation='relu', border_mode='same')(title_emb))
            title_cont_conv.append(Conv1D(128, win_size, activation='relu', border_mode='same')(cont_emb))
        title_cont_conv = merge(title_cont_conv, mode='concat')
        title_cont_pool = GlobalMaxPooling1D()(title_cont_conv)

        # Full connection
        title_cont_features = Dense(1024, activation='relu')(title_cont_pool)

        # Prediction
        preds = Dense(class_num, activation='sigmoid')(title_cont_features)

        self._model = Model([title_input, cont_input], preds)
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
