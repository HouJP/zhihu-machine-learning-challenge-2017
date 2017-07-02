# !/usr/bin/python
# -*- coding:UTF-8 -*-

import json
import sys

from keras import backend as K
from keras.models import load_model

from feature_gen import loadEmbeddingFile, load_validation

reload(sys)
sys.setdefaultencoding('utf8')

def load_map_topic(map_file):
    
    map_topic = {}
    for line in open(map_file):
        line = line.strip('\n')
        if len(line) == 0:
            continue
        part = line.split()
        map_topic[part[1]] = part[0]
    return map_topic

def binary_crossentropy_sum(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
    
def predict(title_x_test, cont_x_test, qid, inverse_dic, bst_model_path, out_name):
#def predict(title_x_test,cont_x_test,bst_model_path,qid,map_topic,out_name,labels_co):
    fout = open(out_name,'w')
    
    model = load_model(bst_model_path)

    model.compile(loss=binary_crossentropy_sum, \
              optimizer='rmsprop', \
              metrics=['accuracy'])
    print "Yesss"
    #model.summary()
    preds = model.predict([title_x_test,cont_x_test], batch_size=512, verbose=1)
    
    print 'preds.shape is ',preds.shape
    
    for i,p in enumerate(preds):
        top = []
        for idx,val in enumerate(p):
            top.append( (idx,val) )    
        top = sorted(top,key = lambda s:s[1],reverse=True)
        #print top[:5]
        
        s = []
        for j in range(5):
            if top[j][0] == 1999:
                continue
            s.append(inverse_dic[str(top[j][0])])
        """    
        tmp = []
        tmp.append(qid[i])
        for j in range(len(top)):
            if top[j][0] == 0:
                print 'label 0 found in '+ str(i)
            if top[j][0] in s:
                tmp.append( map_topic[ str(top[j][0])   ]  )
            if len(tmp) == 6:
                break
        """
    
        fout.write("%s,%s"%(qid[i], ','.join(s)))
        fout.write('\n')
        if i % 10000 == 0:
            print i,' passed.'

if __name__ == '__main__':
    project_pt = '/home/houjianpeng/zhihu-machine-learning-challenge-2017/'
    embedding_index, embedding_matrix = loadEmbeddingFile('%s/data/devel/glove.vec.txt' % project_pt)
    # embedding_index, embedding_matrix = loadEmbeddingFile('./../../Data/raw_data/ieee_zhihu_cup/word_embedding.txt')
    global index_dic 
    index_dic = embedding_index
    title_x_test, cont_x_test, y_test, qid = load_validation('./../../data/train_data/title_content_word.test.csv', embedding_index)
    with open("/home/houjianpeng/zhihu-machine-learning-challenge-2017/data/devel/InverseHashLabel.dic", "r") as fin:
        inverse_dic = json.load(fin)
    predict(title_x_test, cont_x_test, qid, inverse_dic, '/home/houjianpeng/zhihu-machine-learning-challenge-2017/data/model/cnn.title-cont.sum.finetune.model-50w.h5.27.round', "pred.csv")

    #qid = load_test_qid('./ieee_zhihu_cup/question_eval_set.txt')
    #map_topic = load_map_topic('./ieee_zhihu_cup/map_topic.out')
    
    #title_x_test,cont_x_test = load_test('./feature/test_format_out')    
    #title_x_test,cont_x_test = load_test('./feature/test_title_cont_out')    
    
    #labels_co = load_labels_co('ieee_zhihu_cup/train_labels_co.txt')
    #predict(title_x_test, cont_x_test, './model/cnn.title-cont.sum.finetune.model-50w.h5.20.round',qid,map_topic,'./out/pred.sum.finetune.title-cont.20.round.csv',labels_co)
