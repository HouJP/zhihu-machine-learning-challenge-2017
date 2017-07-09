#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 10:31
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model, model_from_json

from bin.utils import LogUtil
from loss import binary_crossentropy_sum


class TitleContentCNN(object):
    def __init__(self,
                 title_word_length,
                 content_word_length,
                 title_char_length,
                 content_char_length,
                 class_num,
                 word_embedding_matrix,
                 char_embedding_matrix,
                 btm_embedding_matrix,
                 optimizer,
                 metrics):
        # set attributes
        self.title_word_length = title_word_length
        self.content_word_length = content_word_length
        self.title_char_length = title_char_length
        self.content_char_length = content_char_length
        self.class_num = class_num
        self.word_embedding_matrix = word_embedding_matrix
        self.char_embedding_matrix = char_embedding_matrix
        self.btm_embedding_matrix = btm_embedding_matrix
        self.optimizer = optimizer
        self.metrics = metrics
        # Placeholder for input (title and content)
        title_word_input = Input(shape=(title_word_length,), dtype='int32', name="title_word_input")
        cont_word_input = Input(shape=(content_word_length,), dtype='int32', name="content_word_input")

        title_char_input = Input(shape=(title_char_length,), dtype='int32', name="title_char_input")
        cont_char_input = Input(shape=(content_char_length,), dtype='int32', name="content_char_input")

        btm_input = Input(shape=(1,), dtype='int32', name="btm_input")

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

        btm_embedding_layer = Embedding(len(btm_embedding_matrix),
                                        100,
                                        weights=[btm_embedding_matrix],
                                        trainable=True, name='btm_embedding')
        btm_emb = btm_embedding_layer(btm_input)

        # Create a convolution + max pooling layer
        title_content_features = list()
        for win_size in range(2, 6):
            # batch_size x doc_len x embed_size
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(128, win_size, activation='relu', padding='same')(title_word_emb)))
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(128, win_size, activation='relu', padding='same')(cont_word_emb)))
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(128, win_size, activation='relu', padding='same')(title_char_emb)))
            title_content_features.append(
                GlobalMaxPooling1D()(Conv1D(128, win_size, activation='relu', padding='same')(cont_char_emb)))
        title_content_features.append(
            GlobalMaxPooling1D()(Conv1D(128, 1, activation='relu', padding='same')(btm_emb)))
        title_content_features = concatenate(title_content_features)

        # Full connection
        title_cont_features = Dense(1024, activation='relu')(title_content_features)

        # Prediction
        preds = Dense(class_num, activation='sigmoid')(title_cont_features)

        self._model = Model([title_word_input, cont_word_input, title_char_input, cont_char_input, btm_input], preds)
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
        # compile model
        self._model.compile(loss=binary_crossentropy_sum, optimizer=self.optimizer, metrics=self.metrics)
        self._model.summary()
        LogUtil.log('INFO', 'load model (%s) from disk done' % model_fp)

    def fit(self, x, y, batch_size=32, epochs=1, validation_data=None):
        self._model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, x, batch_size, verbose):
        return self._model.predict(x, batch_size=batch_size, verbose=verbose)
