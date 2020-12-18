# coding:utf-8
from __future__ import division
import numpy as np
import re

#####################################  分割线  ####################################
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

def senAnalysis(model,senModel,clauseVec):
    devMat=np.array(transform_to_matrix(clauseVec,model))
    devMat = devMat.reshape(devMat.shape[0],  devMat.shape[1], devMat.shape[2], 1)
    devMat = devMat.astype('float32')
    result=senModel.predict(devMat,verbose=0)
    return result[0][1]

def opinionMining(model,senModel,featureSet,clauseVec):
    opinionSet={}
    for clause in clauseVec:
        featureContain=list(set([featureSet[fea] for fea in featureSet.keys() if fea in clause]))
        if len(featureContain)==0:continue
        predict=senAnalysis(model,senModel,[clause])
        for feature in featureContain:
            if not feature in opinionSet.keys():
                opinionSet[feature]=[]
            opinionSet[feature].append(predict)
    for key in opinionSet.keys():
        opinionSet[key]=sum(opinionSet[key])/len(opinionSet[key])
    return opinionSet

import gensim
from keras.models import load_model
import jieba
jieba.load_userdict('model/userdict.txt')
#载入产品特征集合
def loadFeatureSet():
    featureSet={}
    f=open('model/featureSet.txt',encoding='utf-8')
    for row in f.readlines()[1:]:
        data=row.replace('\n','').split('\t')
        featureSet[data[0]]=data[1]
    return featureSet
featureSet=loadFeatureSet()
model=gensim.models.Word2Vec.load('model/word2vec/carCommentData.model')
senModel=load_model('model/sentimentAnalysis.h5')

commentSet='''保时捷提供动能回收,所以在电耗和油耗上都很满意而且已经这个价位的车谈省电或省油不如谈省心,已经被召回几次了,希望不会再通知我交出钥匙。保时捷售后远比其它品牌好。被召回过两次,一次是散热风扇电线靠着导热板,还有一次更换前桥控制臂。摒弃通病不谈没什么好吐槽的地方。关键词:稳定性。就是这种夸张的稳定性让918变成非常容易控制的车,细节上比如强悍的提速感,pdk换档没有闯动感,地板油时动机和电机同时工作这么强的扭矩,高效的动力输出,车轮竟然没打滑。918是我驾驶过提速最夸张的车,就算是布加迪威龙也占不了便宜。强大的抓地力,出色的空气动力,完全就是合法上路的赛车。实在太贵了!保时捷绝版的旗舰车型它的价值远高于发售价格,相信它的实力更强大但我无法全力驾驶它。还有1.75吨的重量还是会让你感觉非常大的惯性,没有不满意算是918的软肋吧。非要鸡蛋里挑骨头而言,太稳定。918车上先进的设备会主动辅助驾驶,帮你控制稳定车辆,不会让你越界,如同欧洲的老管家,足智多谋却没有激情。太太168cm暂时我们还没有baby,带宠物出也会选择其它车辆代步,918成了我们偶尔约会享受人世界的选择。上下班和太太同行我的公文包就需要太太座椅靠前放置在副驾驶后面了。储物空间:副驾驶前的手套箱成了车内最大的储物空间,太太的小包包会习惯放置在中控下面,把手前有小储物格放小瓶饮料还是可以的,因为是硬顶敞篷的关系,太阳镜变得很尴尬,我收纳在手套箱内,要是驾驶途中想取就有些麻烦。放行李:非敞篷情况放行李完全没问题,鱼和熊掌不可兼得。乘坐空间:身高183cm坐直后头部空余指,不顶头肢体活动不会太压抑,已非常满足,驾驶时间长后,脖子酸疼。横向对比:同样是中置引擎458italia座椅舒适度上优于918,其它方面完败458italia。亚马逊tgt测试结果918加速比布加迪还快。1280牛米的扭矩换谁脑袋都发麻!憋问我极速,我怂,特别怂!跑高速:凌晨点钟试过高速全长2.5km隧道时速260km,感觉脑子被丢在后面。敞篷状态还没试过,有机会试试。加速表现:运动模式下可玩性还是非常高的,pdk切换档很快。保时捷提供了多重驾驶模式,纯电动模式,保时捷说可以驾驶30km,对我而言是在回家时以特斯拉方式的进车库,在些路段场合允许安静的穿行。混合动力模式,在电机和动机之间智能选择最经济实惠的驱动方式,达不到媒体宣传升百公里,和驾驶习惯有关但比五菱宏光还省油。运动模式,解禁875匹马力4.6V8发动机,每个红绿灯都是起跑线。赛道模式,点油就炸,平时完全不敢开启,犹如旋涡鸣人体内的九尾,就是妖兽。通病不谈,谁也不会开着跑长途,座椅很舒服包裹性也很好就是时间长了腰不舒服,脖子酸疼…低调,德国人外观设计确实没有意大利妖媚,918提车月后和母上大人喝下午茶,路过辆蓝色的911targa4,训话到人家颜色好看,不像我灰不溜秋。外观而言在LaFerrari面前自愧不如。选择石榴红加银色装饰,并没选黑柠,选黑柠如果不加碳纤维装饰感觉不完美,那问题来了碳纤维装饰要加价,杯托碳纤维加价,这都加了吧座椅换真皮加价,不如直接上WeissachPackage…加价无止境呀骚年。仪表中控:中控的功能之多至今还在研究,等太太时可以上网打发无聊,忘了称赞保时捷配置了11个扬声器的音响,放着音乐上上网很满足了。驾驶中用触控感觉不好,对旋钮和按键有好感,功能习惯后就不吹毛求疵了。时至今日依然很满意,很多朋友问为何不考虑p1,订车前有很认真的对比过,因为价格和定位相同,但918车顶可以掀开,太太和我直都想拥有辆敞篷车不负海滨之城。电动调节座椅(p1座椅调节需要扳手?),动能回收(回收刹车能量,并且将能量转化储存在电池)。十而立,在而立之年前给自己份完美的礼物。驾驶过carreragt、enzo,法拉利限制要求必须拥有在售旗舰车型才具备购买资格而且定价也是无力承担,918的定位和价格成了唯的选择,武装到牙齿的超级跑车。'''
clauseSet=clauseSegmentation(commentSet)
clauseVec=[list(jieba.cut(clause.upper())) for clause in clauseSet]
opinionSet=opinionMining(model,senModel,featureSet,clauseVec)
print(opinionSet)





