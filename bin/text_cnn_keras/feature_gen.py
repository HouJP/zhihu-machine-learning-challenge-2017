# !/usr/bin/python
# -*- coding:UTF-8 -*-

import sys
import time
import numpy as np
from utils import LogUtil


reload(sys)
sys.setdefaultencoding('utf8')

def load_validation(val_file, index_dic, size=200):
    qid = []    
    title_x_val = []
    cont_x_val = []
    y_val = []
    
    for line in open(val_file):
        line = line.strip('\n')
        part = line.split("\t")
        assert(len(part) == 4)
        
        qid.append(part[0])

        title_word = [index_dic[x] for x in part[1].split(',') if x in index_dic]
        title_word = title_word + [0] * (size-len(title_word)) if len(title_word) < size else title_word[:200]
        cont_word = [index_dic[x] for x in part[2].split(',') if x in index_dic]
        cont_word = cont_word + [0] * (size-len(cont_word)) if len(cont_word) < size else cont_word[:200]
        title_x_val.append(title_word)
        cont_x_val.append(cont_word)
        
        tmp = [0] * 2000
        for t in part[3].split(','):
            if t == "":
                continue
            tmp[int(t)] = 1
        y_val.append(tmp)
    
    title_x_val = np.asarray(title_x_val,dtype='int32')
    cont_x_val = np.asarray(cont_x_val,dtype='int32')
    y_val = np.asarray(y_val,dtype='int32')
    
    print 'title_x_val.shape is ',title_x_val.shape
    print 'cont_x_val.shape is ',cont_x_val.shape
    print 'y_val.shape is ',y_val.shape
    return title_x_val, cont_x_val, y_val, qid

def loadEmbeddingFile(embeddingfile):
    embedding_size = 0
    word_num = 0
    
    for line in open(embeddingfile):
        values = line.split()
        if len(values) == 0:
            continue
        word_num += 1
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_size = len(coefs)    
    
    print "word_num is ",word_num
    print "embedding size is ",embedding_size
    
    embedding_index = {}
    embedding_matrix = np.zeros( (word_num + 1, embedding_size) )  
    
    cnt = 0
    for line in open(embeddingfile):
        values = line.split()
        if len(values) == 0:
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        cnt += 1
        embedding_index[word] = cnt
        embedding_matrix[cnt] = coefs
    
    print ("Total embedding word is %s ." % len(embedding_index) )
    
    return embedding_index, embedding_matrix

def load_label(train_labels_file):
    
    instance_num = 0
    for line in open(train_labels_file):
        line = line.strip('\n')
        if len(line) == 0:
            continue
        instance_num += 1
    
    print "instance_num is ",instance_num
    
    labels = np.zeros((instance_num,20))
    #labels = []
    cnt = 0
    for line in open(train_labels_file):
        line = line.strip('\n')
        if len(line) == 0:
            continue
        part = line.split()
        assert(len(part) == 2)
        part = part[1].split(',')
        
        for idx,p in enumerate(part):
            labels[cnt][idx] = int(p)
        
        cnt += 1
    print labels[:2]
    labels = np.asarray(labels,dtype='int32')
    labels = labels.astype(np.string_)
    
    print labels[:2]
    print "labels.shape is",labels.shape
    return labels
        
def formatData(embedding_index,in_file):
    
    #only word
    title_doc_vec = []
    cont_doc_vec = []
    
    for line in open(in_file):
        line = line.strip('\n')
        if len(line) == 0:
            continue
        title_q_vec = [0 for i in range(200)]
        cont_q_vec = [0 for i in range(200)]

        part = line.split()
        
        #title_part
        if len(part) >= 3:
            title_words = part[2].split(',')
            for index,w in enumerate(title_words):
                if embedding_index.has_key(w):
                    title_q_vec[index] = embedding_index[w]

        title_doc_vec.append(title_q_vec)
        
        #cont part
        if len(part) >= 5:
            cont_words = part[4].split(',')
            for index,w in enumerate(cont_words):
                if index >= 200:
                    break
                if embedding_index.has_key(w):
                    cont_q_vec[index] = embedding_index[w]
                    
        cont_doc_vec.append(cont_q_vec)

    print "Word Title Doc len is",len(title_doc_vec)
    print title_doc_vec[:2]
    title_data = np.asarray(title_doc_vec,dtype='int32')
    title_data = title_data.astype(np.string_)
    
    print title_data[:2]
    print 'Word title_data.shape is',title_data.shape

    print "Word Cont Doc len is",len(cont_doc_vec)
    print cont_doc_vec[:2]
    cont_data = np.asarray(cont_doc_vec,dtype='int32')
    cont_data = cont_data.astype(np.string_)
    
    print cont_data[:2]
    print 'Word cont_data.shape is',cont_data.shape

    return title_data,cont_data

def formatDataChar(embedding_index,in_file):
    
    #only char
    title_doc_vec = []
    cont_doc_vec = []
    
    for line in open(in_file):
        line = line.strip('\n')
        if len(line) == 0:
            continue
        title_q_vec = [0 for i in range(300)]
        cont_q_vec = [0 for i in range(300)]

        part = line.split()
        
        #title_part
        if len(part) >= 2:
            title_words = part[1].split(',')
            for index,w in enumerate(title_words):
                if embedding_index.has_key(w):
                    title_q_vec[index] = embedding_index[w]

        title_doc_vec.append(title_q_vec)
        
        #here    
        #cont part
        if len(part) >= 4:
            cont_words = part[3].split(',')
            for index,w in enumerate(cont_words[-300:]):
                if index >= 300:
                    break
                if embedding_index.has_key(w):
                    cont_q_vec[index] = embedding_index[w]
                    
        cont_doc_vec.append(cont_q_vec)

    print "Char Title Doc len is",len(title_doc_vec)
    print title_doc_vec[:2]
    title_data = np.asarray(title_doc_vec,dtype='int32')
    title_data = title_data.astype(np.string_)
    
    print title_data[:2]
    print 'Char title_data.shape is',title_data.shape

    print "Char Cont Doc len is",len(cont_doc_vec)
    print cont_doc_vec[:2]
    cont_data = np.asarray(cont_doc_vec,dtype='int32')
    cont_data = cont_data.astype(np.string_)
    
    print cont_data[:2]
    print 'Char cont_data.shape is',cont_data.shape

    return title_data,cont_data

def front0padding_convert(back0padding_list):
    
    first0 = -1
    for idx,b in enumerate(back0padding_list):
        if b == '0':
            first0 = idx
            break
    if first0 == -1:
        return back0padding_list
    #print 'first0 is ',first0
    non0part = back0padding_list[:first0]
    
    #print non0part
    zeropart = ['0' for i in range(len(back0padding_list) - len(non0part))]
    
    zeropart.extend(non0part)
    
    return zeropart

def shuffle_data(word_title_data_train,word_cont_data_train,char_title_data_train,char_cont_data_train,labels):
    
    print "data size is ",word_title_data_train.shape[0]
    indices = np.arange(word_title_data_train.shape[0])
    np.random.shuffle(indices)
    
    word_title_data_train = word_title_data_train[indices]
    word_cont_data_train = word_cont_data_train[indices]
    
    char_title_data_train = char_title_data_train[indices]
    char_cont_data_train = char_cont_data_train[indices]
    

    labels = labels[indices]
       
    #nb_validation_samples = int(0.2 * word_title_data_train.shape[0])
    nb_validation_samples = 100000
        
    word_title_x_train = word_title_data_train[:-nb_validation_samples]
    word_cont_x_train = word_cont_data_train[:-nb_validation_samples]
    
    char_title_x_train = char_title_data_train[:-nb_validation_samples]
    char_cont_x_train = char_cont_data_train[:-nb_validation_samples]
    
    y_train = labels[:-nb_validation_samples]

    word_title_x_val = word_title_data_train[-nb_validation_samples:]
    word_cont_x_val = word_cont_data_train[-nb_validation_samples:]
        
    char_title_x_val = char_title_data_train[-nb_validation_samples:]
    char_cont_x_val = char_cont_data_train[-nb_validation_samples:]
        
    y_val = labels[-nb_validation_samples:]
    
    train_out = open('../feature/train_title_cont_word_char_out_0paddingback','w')
    for i in range(len(word_title_x_train)):
        train_out.write( ','.join(list(word_title_x_train[i])) +'\t'   )
        train_out.write( ','.join(list(word_cont_x_train[i])) +'\t'   )
        
        train_out.write( ','.join(list(char_title_x_train[i])) +'\t'   )
        train_out.write( ','.join(list(char_cont_x_train[i])) +'\t'   )
        

        train_out.write( ','.join( list(y_train[i])  )    )
        train_out.write('\n')
    
    val_out = open('../feature/val_title_cont_word_char_out_0paddingback','w')
    for i in range(len(word_title_x_val)):
        val_out.write( ','.join(list(word_title_x_val[i])) + '\t'  )
        val_out.write( ','.join(list(word_cont_x_val[i])) + '\t'  )
    
        val_out.write( ','.join(list(char_title_x_val[i])) + '\t'  )
        val_out.write( ','.join(list(char_cont_x_val[i])) + '\t'  )
        
        val_out.write( ','.join(list(y_val[i])  )    )
        val_out.write('\n')
    
    #front 0 padding part
    train_out = open('../feature/train_title_cont_word_char_out_0paddingfront','w')
    for i in range(len(word_title_x_train)):
        train_out.write( ','.join( front0padding_convert(list(word_title_x_train[i]) )) +'\t'   )
        train_out.write( ','.join( front0padding_convert(list(word_cont_x_train[i]) )) +'\t'   )
        
        train_out.write( ','.join( front0padding_convert(list(char_title_x_train[i]))) +'\t'   )
        train_out.write( ','.join( front0padding_convert(list(char_cont_x_train[i]))) +'\t'   )
        

        train_out.write( ','.join( list(y_train[i])  )    )
        train_out.write('\n')
    
    val_out = open('../feature/val_title_cont_word_char_out_0paddingfront','w')
    for i in range(len(word_title_x_val)):
        val_out.write( ','.join( front0padding_convert(list(word_title_x_val[i]))) + '\t'  )
        val_out.write( ','.join( front0padding_convert(list(word_cont_x_val[i]))) + '\t'  )
    
        val_out.write( ','.join( front0padding_convert(list(char_title_x_val[i]))) + '\t'  )
        val_out.write( ','.join( front0padding_convert(list(char_cont_x_val[i]))) + '\t'  )
        
        val_out.write( ','.join(list(y_val[i])  )    )
        val_out.write('\n')

def out_test(word_title_data_test,word_cont_data_test,char_title_data_test,char_cont_data_test):
    
    print 'title_data_test.shape is ',word_title_data_test.shape
    
    test_out = open('../feature/test_title_cont_word_char_out_0paddingback','w')
    for i in range(len(word_title_data_test)):
        test_out.write( ','.join(list(word_title_data_test[i])) + '\t'  )
        test_out.write( ','.join(list(word_cont_data_test[i])) + '\t'  )
        
        test_out.write( ','.join(list(char_title_data_test[i])) + '\t'  )
        test_out.write( ','.join(list(char_cont_data_test[i])) + '\t'  )

        test_out.write('\n')

    test_out = open('../feature/test_title_cont_word_char_out_0paddingfront','w')
    for i in range(len(word_title_data_test)):
        test_out.write( ','.join(front0padding_convert(list(word_title_data_test[i]))) + '\t'  )
        test_out.write( ','.join(front0padding_convert(list(word_cont_data_test[i]))) + '\t'  )
        
        test_out.write( ','.join(front0padding_convert(list(char_title_data_test[i]))) + '\t'  )
        test_out.write( ','.join(front0padding_convert(list(char_cont_data_test[i]))) + '\t'  )

        test_out.write('\n')

if __name__ == '__main__':
    word_embedding_index,word_embedding_matrix = loadEmbeddingFile('../ieee_zhihu_cup/word_embedding.txt')
    char_embedding_index,char_embedding_matrix = loadEmbeddingFile('../ieee_zhihu_cup/char_embedding.txt')
    
    labels = load_label('../ieee_zhihu_cup/train_labels.txt')
    
    word_title_data_train,word_cont_data_train = formatData(word_embedding_index,'../ieee_zhihu_cup/question_train_set.txt')
    word_title_data_test,word_cont_data_test = formatData(word_embedding_index,'../ieee_zhihu_cup/question_eval_set.txt')
    
    char_title_data_train,char_cont_data_train = formatDataChar(char_embedding_index,'../ieee_zhihu_cup/question_train_set.txt')
    char_title_data_test,char_cont_data_test = formatDataChar(char_embedding_index,'../ieee_zhihu_cup/question_eval_set.txt')
    
    shuffle_data(word_title_data_train,word_cont_data_train,char_title_data_train,char_cont_data_train,labels)
    out_test(word_title_data_test,word_cont_data_test,char_title_data_test,char_cont_data_test)
