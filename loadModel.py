# coding:utf-8
from __future__ import division
import numpy as np
import jieba
import jieba.posseg as pseg
jieba.load_userdict('model/userdict.txt')
import os
import time
import gensim
from keras.models import load_model

#为输入文本构建词向量
def create_vector(sen,padding_size,vec_size):
    matrix=[]
    for i in range(padding_size):
        try:
            matrix.append(model[sen[i]].tolist())
        except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
            matrix.append([0] * vec_size)
    return matrix

def transform_to_matrix(x,model, padding_size=128, vec_size=128):
    res = []
    for sen in x:
        matrix =create_vector(sen,padding_size,vec_size)  
        res.append(matrix)
    return res

def trans(testLabel):
    newLabel=[]
    for i in range(len(testLabel)):
        if testLabel[i][0]>testLabel[i][1]:
            newLabel.append(0)
        else:
            newLabel.append(1)
    return newLabel

comment='还有脚刹，很不习惯，老是感觉不太安全，还有就是没有优惠。'
devComment=[[w for w in jieba.cut(comment)]]

model=gensim.models.Word2Vec.load('model/word2vec/carCommentData.model')

devMat = np.array(transform_to_matrix(devComment,model))
devMat = devMat.reshape(devMat.shape[0],  devMat.shape[1], devMat.shape[2], 1)
devMat = devMat.astype('float32')

senModel=load_model('model/sentimentAnalysis.h5')
result=senModel.predict(devMat,verbose=0)
predict=trans(result)

print(predict)
if predict[0]==1:
    print('positive')
if predict[0]==0:
    print('negative')
