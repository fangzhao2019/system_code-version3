# coding:utf-8
import pymysql
import opinionMining
import numpy as np
from pymongo import MongoClient
import jieba
jieba.load_userdict('model/userdict.txt')
import re
import gensim
from keras.models import load_model

#载入产品特征集合
def loadFeatureSet(filename):
    featureSet={}
    f=open(filename,encoding='utf-8')
    for row in f.readlines()[1:]:
        data=row.replace('\n','').split('\t')
        featureSet[data[0]]=data[1]
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
    reg9=re.compile(u'[，,]+')
    reg10=re.compile(u'\t+')
    comment=reg1.sub(u'。【Instead】',comment)
    comment=reg2.sub(u'？【Instead】',comment)
    comment=reg3.sub(u'！【Instead】',comment)
    comment=reg4.sub(u'；【Instead】',comment)
    comment=reg5.sub(u'【Instead】',comment)
    comment=reg6.sub(u'……【Instead】',comment)
    comment=reg7.sub(u'【Instead】',comment)
    comment=reg9.sub('，【Instead】',comment)
    comment=reg10.sub('【Instead】',comment)
    clauseSet=re.split(u'【Instead】',comment)
    clauseList=[clause for clause in clauseSet if len(clause)>5]
    return clauseList

#情感分析
def create_vector(sen,padding_size,vec_size,word2VecModel):
    matrix=[]
    for i in range(padding_size):
        try:
            matrix.append(word2VecModel[sen[i]].tolist())
        except:
            matrix.append([0] * vec_size)
    return matrix

def transform_to_matrix(x,word2VecModel, padding_size=128, vec_size=128):
    res = []
    for sen in x:
        matrix =create_vector(sen,padding_size,vec_size,word2VecModel)  
        res.append(matrix)
    return res

def senAnalysis(word2VecModel,senModel,clause):
    devMat=np.array(transform_to_matrix(clause,word2VecModel))
    devMat = devMat.reshape(devMat.shape[0],  devMat.shape[1], devMat.shape[2], 1)
    devMat = devMat.astype('float32')
    result=senModel.predict(devMat,verbose=0)
    return result[0][1]

def featureSentiment(word2VecModel,senModel,featureSet,clauseVec):
    opinionSet={}
    for clause in clauseVec:
        featureContain=list(set([featureSet[fea] for fea in featureSet.keys() if fea in clause]))
        if len(featureContain)==0:continue
        predict=senAnalysis(word2VecModel,senModel,[clause])
        for feature in featureContain:
            if not feature in opinionSet.keys():
                opinionSet[feature]=[]
            opinionSet[feature].append(predict)
    for key in opinionSet.keys():
        opinionSet[key]=sum(opinionSet[key])/len(opinionSet[key])
    return opinionSet

def splitDataSet(dataSet, axis, value):#按照给定特征划分数据集
    retDataSet = []#用来存储划分结果的列表
    for featVec in dataSet:#对于每行数据
        if featVec[axis] == value:#如果选择的特征等于给定值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)#将这行数据存入划分结果的列表中（不包含给定特征）
    return retDataSet#返回划分结果数据集

def formatDataSet(opinionSet):
    formatOpinionSet={}
    feaUniqueVals=set([exa[0] for exa in opinionSet])
    for value1 in feaUniqueVals:
        subDataSet=splitDataSet(opinionSet,0,value1)
        senUniqueVals=set([exa[0] for exa in subDataSet])
        subDic={}
        for value2 in senUniqueVals:
            commentSet=[c[0] for c in splitDataSet(subDataSet,0,value2)]
            subDic[value2]=commentSet
        keys=subDic.keys()
        if '' in keys:
            if len(keys)>1:
                subDic.pop('')
            if len(keys)==1:
                subDic={}
        formatOpinionSet[value1]=subDic
    return formatOpinionSet

def countFeatureNumber(featureSet,commentVec):
    featureCount2={}
    for fea in featureSet.keys():
        for vec in commentVec:
            if fea==vec:
                if not fea in featureCount2.keys():
                    featureCount2[fea]=0
                featureCount2[fea]+=1
    featureCount1={}
    for key in featureCount2.keys():
        bigFea=featureSet[key]
        if not bigFea in featureCount1.keys():
            featureCount1[bigFea]=0
        featureCount1[bigFea]+=featureCount2[key]
    return featureCount1,featureCount2

def countOpinion(opinionSet,featureSet):
    opinionSetCount={}
    for key1 in opinionSet.keys():
        bigFea=featureSet[key1]
        if not bigFea in opinionSetCount.keys():
            opinionSetCount[bigFea]={}
        for key2 in opinionSet[key1].keys():
            fe_op='%s%s'%(key1,key2)
            if not fe_op in featureOpinion:
                continue
            sen=opinionSet[key1][key2]
            opinionSetCount[bigFea][fe_op]=sen
    return opinionSetCount
            

############################数据处理############################

client=MongoClient('192.168.1.101',30000)
collection1=client.car_autohome.comment
collection2=client.car_dev.autohome_comment_processed_new

featureSet=loadFeatureSet('model/featureSet.txt')
#######
filterFeature=loadFeatureSet('model/filterFeature.txt')
featureOpinion=list(loadFeatureSet('model/featureOpinionSet.txt').keys())
#######
senWords=[w.replace('\n','') for w in open('model/senWords.txt',encoding='utf-8').readlines()[1:]]
degreeWords=[w.replace('\n','') for w in open('model/degreeWords.txt',encoding='utf-8').readlines()[1:]]
notWords=[w.replace('\n','') for w in open('model/notWords.txt',encoding='utf-8').readlines()[1:]]

print('一共包含待处理口碑数据%s条'%collection1.find().count())
print('一共包含产品特征%d个'%len(featureSet))
print('一共包含情感词%d个'%len(senWords))
word2VecModel=gensim.models.Word2Vec.load('model/word2vec/carCommentData.model')
senModel=load_model('model/sentimentAnalysis.h5')
print('成功载入词向量模型与情感分析模型')

i=0
for row in collection1.find({"status":3}):
    i+=1
    if i%100==0:print(i)
    new_mongoData={}
    try:
        data=row['data']
    except:
        continue
    #处理评论数据
    if '评价' in data.keys():
        comment=data['评价']
        commentSet='【Instead】'.join(list(comment.values()))
        clauseSet=clauseSegmentation(commentSet)

        commentVec=list(jieba.cut(commentSet.replace('【Instead】','')))
        clauseVec=[list(jieba.cut(clause.upper())) for clause in clauseSet]
    else:
        continue
    ########################################################
    #满意度（一级特征）
    featureSatisfaction=featureSentiment(word2VecModel,senModel,filterFeature,clauseVec)
    new_mongoData['满意度']=str(featureSatisfaction)

    #关注度
    featureCount1,featureCount2=countFeatureNumber(featureSet,commentVec)
    new_mongoData['一级特征关注度']=featureCount1
    new_mongoData['二级特征关注度']=featureCount2

    #特征观点提取（二级特征）
    opinionSet=opinionMining.opinionMining(commentSet.replace('【Instead】',''),list(featureSet.keys()),senWords,degreeWords,notWords)
    opinionSetCount=countOpinion(opinionSet,featureSet)
    new_mongoData['特征观点']=opinionSetCount
    
    try:
        collection2.insert(new_mongoData)
    except:
        continue

    
