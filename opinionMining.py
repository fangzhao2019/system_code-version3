# coding:utf-8
from __future__ import division
from pymongo import MongoClient
import jieba
import jieba.posseg as pseg
from openpyxl import load_workbook
from openpyxl import Workbook
import re
jieba.load_userdict('model/userdict.txt')

#####################################  分割线  #####################################

#载入产品特征词以及观点词
def load_feature_senword_dataSet():
    featureSet={}
    f=open('model/featureSet.txt',encoding='utf-8')
    for row in f.readlines()[1:]:
        data=row.replace('\n','').split('\t')
        featureSet[data[0]]=data[1]
    senWords=[w.replace('\n','') for w in open('model/senWords.txt',encoding='utf-8').readlines()[1:]]
    degreeWords=[w.replace('\n','') for w in open('model/degreeWords.txt',encoding='utf-8').readlines()[1:]]
    notWords=[w.replace('\n','') for w in open('model/notWords.txt',encoding='utf-8').readlines()[1:]]

    return list(featureSet.keys()),senWords,degreeWords,notWords

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
    clauseList=[clause for clause in clauseSet if len(clause)>2]
    return clauseList
#找到某个特征词对应的所有位置
def myfind(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

def deepAnalysis(f_index,index2,senword,degwordSet,notwordSet,clauseVecSet):
    opinion=''
    degword=''
    notword=''
    degIndex=0
    notIndex=0
    for x in range(f_index+1,index2):
        if clauseVecSet[x][0] in degwordSet:
            degword=clauseVecSet[x][0]
            degIndex=x
        if clauseVecSet[x][0] in notwordSet:
            notword=clauseVecSet[x][0]
            notIndex=x
    if degIndex>notIndex:
        opinion=notword+degword+senword
    if notIndex>=degIndex:
        opinion=degword+notword+senword
    return opinion

#从左边找，停止条件：1.遇到逗号；2.抵达句子开始；3.遇到其它特征词;4.不需要程度词和否定词
def forwardSearch(f_index,featureSet,senwordSet,clauseVecSet):
    opinion=''
    for k in range(1,6):
        
        index1=f_index-k
        if index1>=0:
            if clauseVecSet[index1][1] == 'x':break
            if clauseVecSet[index1][0] in featureSet:break
            #if (clauseVecSet[index1][0] in senwordSet) or (clauseVecSet[index1][1] =='a'):
            if clauseVecSet[index1][0] in senwordSet:
                opinion=clauseVecSet[index1][0]
                break
    return opinion

#从右边找，停止条件：1.遇到逗号；2.抵达句子末端；3.遇到其它特征词；4.需要程度词和否定词
def backwardSearch(f_index,featureSet,senwordSet,degwordSet,notwordSet,clauseVecSet):
    opinion=''
    for k in range(1,6):
        index2=f_index+k
        if index2<len(clauseVecSet):
            if clauseVecSet[index2][1] == 'x':break
            if clauseVecSet[index2][0] in featureSet:break
            #if (clauseVecSet[index2][0] in senwordSet) or (clauseVecSet[index2][1] =='a'):
            if clauseVecSet[index2][0] in senwordSet:
                senword=clauseVecSet[index2][0]
                #定位程度词与否定词
                opinion=deepAnalysis(f_index,index2,senword,degwordSet,notwordSet,clauseVecSet)
                break
    return opinion

def splitDataSet(dataSet, axis, value):#按照给定特征划分数据集
    retDataSet = []#用来存储划分结果的列表
    for featVec in dataSet:#对于每行数据
        if featVec[axis] == value:#如果选择的特征等于给定值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)#将这行数据存入划分结果的列表中（不包含给定特征）
    return retDataSet#返回划分结果数据集

#将数据格式化为字典
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
        formatOpinionSet[value1]=subDic
    return formatOpinionSet

def opinionMining(comment,featureSet,senwordSet,degwordSet,notwordSet):
    #分句
    clauseList=clauseSegmentation(comment)
    opinionSet=[]
    #对于每一条子句，分别提取特征词与观点词
    for i in range(len(clauseList)):
        clause=clauseList[i]
        clauseVecSet=[(w.word,w.flag) for w in pseg.cut(clause)]
        clauseWordSet=[w[0] for w in clauseVecSet]
        #print(clauseVecSet)
        for j in range(len(featureSet)):
            #定位特征词
            feature=featureSet[j]
            f_index=myfind(feature,clauseWordSet)
            if len(f_index)>0:index=f_index[0]
            else:continue
            #定位观点词
            opinion=forwardSearch(index,featureSet,senwordSet,clauseVecSet)
            if len(opinion)==0:
                opinion=backwardSearch(index,featureSet,senwordSet,degwordSet,notwordSet,clauseVecSet)
            if len(opinion)==0:continue
            result=[feature,opinion,clause]
            opinionSet.append(result)
    formatOpinionSet=formatDataSet(opinionSet)
    return formatOpinionSet

#####################################  分割线  #####################################

featureSet,senwordSet,degwordSet,notwordSet=load_feature_senword_dataSet()
resultSet=opinionMining('内饰太优秀了,外观大气，性价比高！整体感觉良好，再说一遍外观好',featureSet,senwordSet,degwordSet,notwordSet)
print(resultSet)

#clause='外观真的太差了。'
#clauseVecSet=[(w.word,w.flag) for w in pseg.cut(clause)]
#print(clauseVecSet)
#opinion=deepAnalysis(0,2,'太差',degwordSet,notwordSet,clauseVecSet)
#print(opinion)
