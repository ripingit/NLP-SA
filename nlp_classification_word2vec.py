import nltk
import random
from nltk.classify import SklearnClassifier           #nltk中使用scikit-learn
from sklearn.naive_bayes import BernoulliNB           #s-l贝叶斯分类器
from sklearn.ensemble import RandomForestClassifier   #s-l随机森林分类器
from sklearn.neighbors import KNeighborsClassifier    #s-lKNN分类器
from sklearn.svm import SVC                           #s-l支持向量机分类器
from sklearn import metrics                           #计算评价指标
from sklearn import cross_validation                  #划分训练集和测试集 
from nltk.corpus import PlaintextCorpusReader
from gensim.models import word2vec
import numpy as np 
from numpy import *

#加载自己的语料库
print('加载语料库...')
corpus_root_neg=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\neg_pre6000"
corpus_root_pos=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\pos_pre6000"

neg=PlaintextCorpusReader(corpus_root_neg,'.*')
pos=PlaintextCorpusReader(corpus_root_pos,'.*')

documents_neg =[(list(neg.words(fileid)),0)   for fileid in neg.fileids()]
documents_pos =[(list(pos.words(fileid)),1)   for fileid in pos.fileids()]
documents_neg.extend(documents_pos)
documents=documents_neg  
random.shuffle(documents)   #随机打乱

sentences = word2vec.Text8Corpus(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\merge\1000.txt")  #加载词向量训练语料
model = word2vec.Word2Vec(sentences, size=150, min_count=1) #从零开始训练word2vec模型

# 增量训练 
#print('加载模型...')                  
#model =  word2vec.Word2Vec.load(r"E:\Strange\SRTP.11\NLP\data\wx\word2vec_wx") 
#print('增量训练...')   
#model.train(sentences)

# 增量训练效果示例
# mod =  word2vec.Word2Vec.load(r"E:\Strange\SRTP.11\NLP\data\10Gvec\60\Word60.model") 
# mod.similarity(u"不错", u"好")
# sentence=[[u'不错', u'好', u'好'], [u'不错', u'好', u'不错']] 
# mod.train(sentence)
# mod.similarity(u"不错", u"好")
   
# 词向量提取函数 
# a=model['房间']  # 示例            
def document_vecfea(a):
    # a=documents[1][0] # a 为一条评论
    vecfea = {}
    for word in a:
        #如果词汇在模型内，则输出词向量；否则，置为零向量
        try:
            vecfea[word] = model[word] # dict　
        except KeyError:
            continue
           # vecfea[word] = zeros(256)
    b=list(vecfea.values())  # dict
    c=np.array(b) # n * 256维的矩阵
    d=c.mean(axis=0) # 将句子中全体词向量的平均值算作其特征值，d 为这条评论的句向量
    #d=c.sum(axis=0)  # 将句子中全体词向量的和算作其特征值，d 为这条评论的句向量
    return d

print('提取词向量...')
vecfea  = [(document_vecfea(d1), c1) for (d1,c1) in documents] 

x=[d for (d,c) in vecfea] # 可供scikit-learn训练的输入
y=[c for (d,c) in vecfea] # 可供scikit-learn训练的输出（标签）

 
clf0 = KNeighborsClassifier()
clf1 = BernoulliNB()
clf2 = SVC() 


import time
print('KNN交叉验证...') 
start0 = time.clock()
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3) 
elapsed0 = (time.clock() - start0)/3
print('KNNtime:' ,elapsed0)

print('NB交叉验证...') 
start1 = time.clock() 
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3)
elapsed1 = (time.clock() - start1)/3
print('NBtime:' ,elapsed1)

print('SVM交叉验证...') 
start2 = time.clock()  
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3)
elapsed2 = (time.clock() - start2)/3
print('SVMtime:' ,elapsed2)

print ( 'Acc:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean()))


# 交叉验证 F
print('KNN交叉验证...')  
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='f1_macro')  
print('NB交叉验证...')  
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='f1_macro')
print('SVM交叉验证...')  
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='f1_macro')
print ( 'Fscore:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean()))