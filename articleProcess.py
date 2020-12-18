# coding:utf-8
from __future__ import division
import pymysql
import numpy as np
import jieba
import jieba.posseg as pseg
jieba.load_userdict('model/userdict.txt')
import os
import time
import re
import gensim
from keras.models import load_model

#获取产品特征
def loadFeatureSet():
    conn = pymysql.connect(host='47.99.116.136',user='root',passwd='3H1passwd',port=3306,db='car_test',charset='utf8')
    cursor=conn.cursor()
    sql_sentence='select fine_grained_feature_name from fine_grained_featureset'
    cursor.execute(sql_sentence)
    results = cursor.fetchall()
    featureSet=[]
    for r in results:  
        featureSet.append(r[0])
    return featureSet

#将长句切分为短句
def clauseSegmentation(comment):
    reg1=re.compile(u'。+')
    reg2=re.compile(u'[？?]+')
    reg3=re.compile(u'[！!]+')
    reg4=re.compile(u'[;；]+')
    reg5=re.compile(u'\d[、,，：:]+')
    reg6=re.compile(u'…+')
    reg7=re.compile(u'（\d）')
    reg9=re.compile(u'，+')
    reg10=re.compile(u'\t+')
    comment=reg1.sub(u'。【Instead】',comment)
    comment=reg2.sub(u'？【Instead】',comment)
    comment=reg3.sub(u'！【Instead】',comment)
    comment=reg4.sub(u'；【Instead】',comment)
    comment=reg5.sub(u'【Instead】',comment)
    comment=reg6.sub(u'……【Instead】',comment)
    comment=reg7.sub(u'',comment)
    comment=reg9.sub('，',comment)
    comment=reg10.sub('【Instead】',comment)
    clauseSet=re.split(u'【Instead】',comment)
    clauseList=[clause for clause in clauseSet if len(clause)>5]
    clauseVecSet=[list(jieba.cut(clause)) for clause in clauseList]
    return clauseList,clauseVecSet

def featureExtraction(clauseList,clauseVecSet,featureSet):
    formatFeature={}
    for i in range(len(featureSet)):
        feature=featureSet[i]
        for j in range(len(clauseVecSet)):
            clauseVec=clauseVecSet[j]
            clause=clauseList[j]
            if feature in clauseVec:
                if feature not in formatFeature.keys():
                    formatFeature[feature]=[]
                formatFeature[feature].append(clause)
    return formatFeature

def featureCountNumber(formatFeature):
    featureCount={}
    for feature in formatFeature.keys():
        featureCount[feature]=len(formatFeature[feature])
    return featureCount

#情感分析
def create_vector(sen,padding_size,vec_size):
    matrix=[]
    for i in range(padding_size):
        try:
            matrix.append(model[sen[i]].tolist())
        except:
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

#单个句子情感分析
def senAnalysis(model,senModel,clauseVec):
    devMat=np.array(transform_to_matrix(clauseVec,model))
    devMat = devMat.reshape(devMat.shape[0],  devMat.shape[1], devMat.shape[2], 1)
    devMat = devMat.astype('float32')
    result=senModel.predict(devMat,verbose=0)
    predict=trans(result)
    #print(predict)
    return predict[0]

#整篇文章情感分析
def sentimentAnalysis(model,senModel,clauseVecSet):
    senCount=[]
    for clauseVec in clauseVecSet:
        predict=senAnalysis(model,senModel,[clauseVec])#这里非常重要
        senCount.append(predict)
    if len(senCount)==0:
        return 0
    polarity=sum(senCount)/float(len(senCount))
    if polarity>0.5:
        return 1
    else:
        return -1

def dataProcess(model,senModel,featureSet,comment):
    formatData={}
    clauseList,clauseVecSet=clauseSegmentation(comment)
    formatFeature=featureExtraction(clauseList,clauseVecSet,featureSet)
    formatData['featureSet']=formatFeature
    featureCount=featureCountNumber(formatFeature)
    formatData['featureCount']=featureCount
    formatData['polarity']=sentimentAnalysis(model,senModel,clauseVecSet)
    return formatData

#一条评论
#comment='''优点：这款车的外观显得很霸气，尤其那个车灯衬得特别张扬的那种霸气，红色的车身更加出众，'''
#载入模型
#featureSet=loadFeatureSet()
#model=gensim.models.Word2Vec.load('model/word2vec/carCommentData.model')
#senModel=load_model('model/sentimentAnalysis.h5')
#数据处理
#formatData=dataProcess(model,senModel,featureSet,comment)
#print(formatData)
