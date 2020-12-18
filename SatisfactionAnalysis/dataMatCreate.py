import os
import gensim
import time
import numpy as np


def load_initDataSet(filename):
    f=open(filename,encoding='utf-8')
    dataSet=[row.strip() for row in f.readlines()]
    
    cateSet=[]
    commentVecSet=[]
    commentTagSet=[]
    labelSet=[]
    for i in range(1,len(dataSet)):
        data=dataSet[i].split('\t')
        cateSet.append(data[0])
        commentVecSet.append(data[1].split(' '))
        commentTagSet.append(data[2].split(' '))
        labelSet.append([int(d) for d in data[3].split(' ')])
    return cateSet,commentVecSet,commentTagSet,np.array(labelSet)

#为输入文本构建词向量
def create_vector(sen,padding_size,vec_size,model,select):
    matrix=[]
    if select=='Vec':
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0] * vec_size)

    if select=='Tag':
        for i in range(padding_size):
            try:
                vector=[0]*vec_size
                index=model.index(sen[i])
                vector[index]=1
                matrix.append(vector)
            except:
                matrix.append([0]*vec_size)

    if select=='Cate':
        vector=[0]*vec_size
        index=model.index(sen)
        vector[index]=1
        matrix=vector
    return np.array(matrix)  

def transform_to_matrix (x, model, padding_size, vec_size, select):
    res = []
    for sen in x:
        matrix =create_vector(sen,padding_size,vec_size,model,select)  
        res.append(matrix)
    return res

time1=time.time()

POStagSet=open('data/POStag.txt',encoding='utf-8').readlines()[1].strip().split(' ')
categorySet=open('data/category.txt',encoding='utf-8').readlines()[1].strip().split(' ')


trainCategory,trainCommentVec,trainCommentTag,trainLabel=load_initDataSet('data/trainingSet.txt')
print('导入训练数据%d条'%len(trainLabel))

testCategory,testCommentVec,testCommentTag,testLabel=load_initDataSet('data/testingSet.txt')
print('导入测试数据%d条'%len(testLabel))

print('载入词向量模型')
vecmodel=gensim.models.Word2Vec.load('data/word2vec/comment.model')

print('生成训练数据矩阵')
trainMat=transform_to_matrix(trainCommentVec,vecmodel,100,256,'Vec')
trainCate=transform_to_matrix(trainCategory, categorySet, 1, len(categorySet),'Cate')
trainTag=transform_to_matrix(trainCommentTag, POStagSet, 100, len(POStagSet),'Tag')

print('生成测试数据矩阵')
testMat=transform_to_matrix(testCommentVec,vecmodel,100,256,'Vec')
testCate=transform_to_matrix(testCategory, categorySet, 1, len(categorySet),'Cate')
testTag=transform_to_matrix(testCommentTag, POStagSet, 100, len(POStagSet),'Tag')

print(u'数据矩阵构建完毕，正在保存...')
np.save('mat/trainMat.npy',trainMat)
np.save('mat/trainCate.npy',trainCate)
np.save('mat/trainTag.npy',trainTag)
np.save('mat/trainLabel.npy',trainLabel)
np.save('mat/testMat.npy',testMat)
np.save('mat/testCate.npy',testCate)
np.save('mat/testTag.npy',testTag)
np.save('mat/testLabel.npy',testLabel)

time2=time.time()
print('耗时%d秒'%(time2-time1))







