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

#格式化数据来源
def sourceProcess(mongoUrl):
    if 'bitauto' in mongoUrl:
        return '易车网'
    if 'autohome' in mongoUrl:
        return '汽车之家'
    if 'pcauto' in mongoUrl:
        return '太平洋汽车网'
    return ''

#载入车型数据
def loadCarDataSet():
    conn = pymysql.connect(host='47.99.116.136',user='root',passwd='3H1passwd',port=3306,db='car_dev',charset='utf8')
    cursor1=conn.cursor()
    sql_sentence1='''select car_id,car_name from car_new where status=1 or status=2'''
    cursor1.execute(sql_sentence1)
    results1 = cursor1.fetchall()
    carDataSet={}
    for r in results1:
        carDataSet[r[1]]=[]
        cursor2=conn.cursor()
        cursor2.execute('''select model_name from car_new,car_model
                            where car_model.car_id=car_new.car_id
                            and car_model.sell_type<3
                            and car_model.time is not null
                            and car_new.car_id=%s''',r[0])
        results2 = cursor2.fetchall()
        for s in results2:
            carDataSet[r[1]].append(s[0])
    return carDataSet

#载入产品特征集合
def loadFeatureSet():
    featureSet={}
    f=open('model/featureSet.txt',encoding='utf-8')
    for row in f.readlines()[1:]:
        data=row.replace('\n','').split('\t')
        featureSet[data[0]]=data[1]
    return featureSet

#错误匹配规则
def loadErrorModel():
    errorMatch=[]
    f=open('modelErrorMatch.txt',encoding='utf-8')
    for row in f.readlines()[1:]:
        data=row.split()
        errorMatch.append(data)
    return errorMatch

def cosMatch(word1,word2):
    a1=list(set([w for w in word1.replace(' ','')]))
    b1=list(set([w for w in word2.replace(' ','')]))
    c1=sum([1 for w in a1 if w in b1])
    s1=c1/np.sqrt(len(a1)*len(b1))

    a2=word1.split(' ')
    b2=word2.split(' ')
    c2=sum([1 for w in a2 if w in b2])
    s2=c2/np.sqrt(len(a2)*len(b2))

    a3=list(set([word1[i:i+2] for i in range(len(word1)-1)]))
    b3=list(set([word2[i:i+2] for i in range(len(word2)-1)]))
    c3=sum([1 for w in a3 if w in b3])
    s3=c3/np.sqrt(len(a3)*len(b3))

    a4=list(set([word1[i:i+3] for i in range(len(word1)-2)]))
    b4=list(set([word2[i:i+3] for i in range(len(word2)-2)]))
    c4=sum([1 for w in a4 if w in b4])
    s4=c4/np.sqrt(len(a4)*len(b4))
    
    return (s1+s2+s3+s4)/4.

def carProcess(car,carSet):
    for c in carSet:
        if car.upper() == c.upper():
            return c
    else:
        for c in carSet:
            if car.upper() in c.upper():
                return c
            if c.upper() in car.upper():
                return c
    return 0

def modelProcess(carName,model,carDataSet,errorMatch):
    modelSet=carDataSet[carName]
    for m in modelSet:
        if model.upper().replace(' ','') == m.upper().replace(' ',''):
            return m
    modelSet1=[m for m in modelSet]
    #根据典型规则筛选车款
    #...1)年份
    year=model.split()[0]
    for k in modelSet:
        if k.split()[0]!=year:
            modelSet1.remove(k)
    #...2)动力
    modelSet2=[m for m in modelSet1]
    power1=re.findall('\d\.\dT',model)
    if len(power1)>0:
        for k in modelSet1:
            power2=re.findall('\d\.\dT',k)
            if len(power2)>0:
                if power1[0]!=power2[0]:
                    modelSet2.remove(k)
    #...3)容积
    modelSet3=[m for m in modelSet2]
    liter1=re.findall('\d\.\dL',model)
    if len(liter1)>0:
        for k in modelSet2:
            liter2=re.findall('\d\.\dL',k)
            if len(liter2)>0:
                if liter1[0]!=liter2[0]:
                    modelSet3.remove(k)
    #...4)其它规则
    modelSet4=[m for m in modelSet3]
    for k in modelSet3:
        for data in errorMatch:
            if data[0] in model:
                if data[1] in k:
                    modelSet4.remove(k)
                        
    if len(modelSet4)==0:
        return 0
        
    #相似度匹配
    similarity=[]
    for m in modelSet4:
        similar=cosMatch(model.upper(),m.upper())
        similarity.append(similar)
        
    if max(similarity)<0.5:
        return 0
    index=similarity.index(max(similarity))
    finalModel=modelSet4[index]
    return finalModel

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

def mentionCarProcess(clauseVec,featureSet,carSet):
    mentionCarSet=[]
    for clause in clauseVec:
        for car in carSet:
            if car.upper() in clause:
                featureContain=[feature for feature in featureSet.keys() if feature in clause]
                if len(featureContain)==0:
                    mentionCarSet.append([car,'',''])
                else:
                    for feature in featureContain:
                        result=[car,featureSet[feature],''.join(clause)]
                        mentionCarSet.append(result)
    formatMentionCarSet=formatDataSet(mentionCarSet)
    return formatMentionCarSet
############################数据处理############################

client=MongoClient('47.92.211.251',30000)
collection1=client.car_autohome.comment
collection2=client.car_dev.autohome_comment
carDataSet=loadCarDataSet()
carSet=list(carDataSet.keys())
featureSet=loadFeatureSet()
errorMatch=loadErrorModel()
senWords=[w.replace('\n','') for w in open('model/senWords.txt',encoding='utf-8').readlines()[1:]]
degreeWords=[w.replace('\n','') for w in open('model/degreeWords.txt',encoding='utf-8').readlines()[1:]]
notWords=[w.replace('\n','') for w in open('model/notWords.txt',encoding='utf-8').readlines()[1:]]

print('一共包含待处理口碑数据%s条'%collection1.find().count())
print('一共包含车型%d个'%len(carSet))
print('一共包含产品特征%d个'%len(featureSet))
print('一共包含情感词%d个'%len(senWords))
word2VecModel=gensim.models.Word2Vec.load('model/word2vec/carCommentData.model')
senModel=load_model('model/sentimentAnalysis.h5')
print('成功载入词向量模型与情感分析模型')

i=0
for row in collection1.find({"status":3},no_cursor_timeout = True):
    i+=1
    if i%100==0:print(i)
    #if i<130000:continue
    new_mongoData={}
    #id
    new_mongoData['_id']=row['_id']
    
    #dataSource
    new_mongoData['dataSource']=sourceProcess(row['url'])
          
    #modelName
    if 'modelName' in row.keys():
        car=row['modelName']
        carName=carProcess(car,carSet)
        if carName==0:continue
        new_mongoData['carName']=carName
    else:
        continue

    data=row['data']
    #购买车型
    if '购买车型' in data.keys():
        model=data['购买车型']
        modelName=modelProcess(carName,model,carDataSet)
        if modelName==0:continue
        new_mongoData['modelName']=modelName
    elif 'editionName' in row.keys():
        model=row['editionName']
        modelName=modelProcess(carName,model,carDataSet,errorMatch)
        if modelName==0:continue
        new_mongoData['modelName']=modelName
    else:
        continue
        
    #购买地点
    if '购买地点' in data.keys():
        new_mongoData['购买地点']=data['购买地点']
    else:
        new_mongoData['购买地点']=''
        
    #购买时间
    if '购买时间' in data.keys():
        new_mongoData['购买时间']=data['购买时间']
    else:
        new_mongoData['购买时间']=''
        
    #购买价格
    if '裸车购买价' in data.keys():
        new_mongoData['购买价格']=data['裸车购买价']
    else:
        new_mongoData['购买价格']=''

    #处理评论数据
    if '评价' in data.keys():
        comment=data['评价']
        commentSet='【Instead】'.join(list(comment.values()))
        clauseSet=clauseSegmentation(commentSet)

        commentVec=list(jieba.cut(commentSet))
        clauseVec=[list(jieba.cut(clause.upper())) for clause in clauseSet]
    else:
        continue
    
    #满意度（一级特征）
    featureSatisfaction=featureSentiment(word2VecModel,senModel,featureSet,clauseVec)
    new_mongoData['featureSatisfaction']=featureSatisfaction
    
    #关注度（二级特征）
    featureCount=[fea for fea in featureSet.keys() if fea in commentVec]
    new_mongoData['featureCount']=featureCount
    
    #提及车型
    formatMentionCar=mentionCarProcess(clauseVec,featureSet,carSet)
    new_mongoData['formatMentionCar']=formatMentionCar
    
    #形象认知
    imageRecognition=[w for w in senWords if w in commentVec]
    new_mongoData['imageRecognition']=imageRecognition

    #特征观点对提取
    new_commentSet={}
    commentTotal=''#整合全部评论
    commentSet=row['data']['评价']
    whole_featureSet=list(commentSet.keys())
    for i in range(len(whole_featureSet)):
        whole_feature=whole_featureSet[i]
        comment=commentSet[whole_feature]
        commentTotal+=comment
        formatOpinionSet=opinionMining.opinionMining(comment,list(featureSet.keys()),senWords,degreeWords,notWords)
        new_commentSet[whole_feature]=formatOpinionSet
    new_mongoData['formatOpinionSet']=new_commentSet

    try:
        collection2.insert(new_mongoData)
    except:
        continue
