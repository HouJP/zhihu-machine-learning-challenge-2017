# !/usr/bin/python
# -*- coding:UTF-8 -*-

import sys
import time
import gc
import numpy as np
import pandas as pd

from keras import layers
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, merge
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras import backend as K

from utils import LogUtil
from feature_gen import loadEmbeddingFile, load_validation

reload(sys)
sys.setdefaultencoding('utf8')

epoch_sample = 200000

title_x_train = np.zeros((epoch_sample,200))
cont_x_train = np.zeros((epoch_sample,200))
y_train = np.zeros((epoch_sample,2000))
tmp = [0 for i in range(2000)]


project_pt = '/home/houjianpeng/zhihu-machine-learning-challenge-2017/'


def generate_batch_from_file(path, batch_size, size=200):
    
    print "step into generate batch from file."
    
    global title_x_train,cont_x_train,y_train,tmp
    global index_dic
    
    cnt = 0
    
    while 1:
        print 'file open.'
        f = open(path)
        for line in f:
            #x, y = process_line(line)
            line = line.strip('\n')
            if len(line) == 0:
                continue
            part = line.split("\t")
            tmp = [0] * 2000
            for t in part[3].split(','):
                if int(t) != 0:
                    tmp[int(t)] = 1

            title_word = [index_dic[x] for x in part[1].split(',') if x in index_dic]
            title_word = title_word + [0] * (size-len(title_word)) if len(title_word) < size else title_word[:200]
            cont_word = [index_dic[x] for x in part[2].split(',') if x in index_dic]
            cont_word = cont_word + [0] * (size-len(cont_word)) if len(cont_word) < size else cont_word[:200]

            title_x_train[cnt] = np.asarray(title_word, dtype='int32')
            cont_x_train[cnt] = np.asarray(cont_word, dtype='int32')
            y_train[cnt] = np.asarray( tmp,dtype='int32')
            
            cnt += 1    
            if cnt == batch_size:
                yield (title_x_train,cont_x_train, y_train)
                cnt = 0
                #del a
                #gc.collect()
        f.close()
        print 'file close.'

def binary_crossentropy_sum(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def train(embedding_matrix, title_x_val, cont_x_val, y_val):
    """

    :param embedding_matrix: embedding矩阵
    :param title_x_val: 验证集
    :param cont_x_val: 验证集
    :param y_val: 验证集
    :return:
    """
    #model
    title_input = Input(shape=(200,),dtype='int32', name="title_word_input")
    cont_input = Input(shape=(200,),dtype='int32', name="content_word_input")
    
    #trinable = True already
    embedding_layer = Embedding(len(embedding_matrix),\
                    256,\
                    weights=[embedding_matrix],\
                    input_length=200,\
                    trainable=True)
    
    title_sequence_input = embedding_layer(title_input)
    cont_sequence_input = embedding_layer(cont_input)

    title_win_2 = Conv1D(128,2,activation='relu',border_mode='same')(title_sequence_input)
    title_win_3 = Conv1D(128,3,activation='relu',border_mode='same')(title_sequence_input)
    title_win_4 = Conv1D(128,4,activation='relu',border_mode='same')(title_sequence_input)
    title_win_5 = Conv1D(128,5,activation='relu',border_mode='same')(title_sequence_input)

    title_x = merge([title_win_2,title_win_3,title_win_4,title_win_5],mode='concat')
    title_x = GlobalMaxPooling1D()(title_x)
    
    cont_win_2 = Conv1D(128,2,activation='relu',border_mode='same')(cont_sequence_input)
    cont_win_3 = Conv1D(128,3,activation='relu',border_mode='same')(cont_sequence_input)
    cont_win_4 = Conv1D(128,4,activation='relu',border_mode='same')(cont_sequence_input)
    cont_win_5 = Conv1D(128,5,activation='relu',border_mode='same')(cont_sequence_input)

    cont_x = merge([cont_win_2,cont_win_3,cont_win_4,cont_win_5],mode='concat')
    cont_x = GlobalMaxPooling1D()(cont_x)
    
    x = merge([title_x,cont_x],mode='concat')
    x = Dense(1024,activation='relu')(x)
    
    preds = Dense(2000, activation='sigmoid')(x)
    model = Model([title_input,cont_input], preds)
    model.compile(loss=binary_crossentropy_sum,\
            optimizer='rmsprop',\
            metrics=['accuracy'])
    model.summary()

    orde = 0
    for (x1,x2,y) in generate_batch_from_file('%s/data/train_data/title_content_word.train.csv' % project_pt, epoch_sample):
        orde += 1
        print "Round " + str(orde) + " starts."
        model_path = '%s/data/model/cnn.title-cont.sum.finetune.model-50w.h5.' % model + str(orde) + '.round'
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=False, save_weights_only=False)
        model.fit([x1,x2], y, validation_data=( [title_x_val,cont_x_val] , y_val), epochs=1, batch_size=1280,callbacks=[model_checkpoint])

if __name__ == '__main__':
    embedding_index, embedding_matrix = loadEmbeddingFile('%s/data/devel/glove.vec.txt' % project_pt)
    global index_dic 
    index_dic = embedding_index
    title_x_val, cont_x_val, y_val, qid = load_validation('%s/data/train_data/title_content_word.valid.csv' % project_pt, embedding_index)
    train(embedding_matrix,title_x_val,cont_x_val,y_val)

