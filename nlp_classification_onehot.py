import nltk
import random
from nltk.corpus import PlaintextCorpusReader         #加载自定义语料库
from nltk.classify import SklearnClassifier           #nltk中使用scikit-learn
from sklearn.naive_bayes import BernoulliNB           #s-l贝叶斯分类器
from sklearn.ensemble import RandomForestClassifier   #s-l随机森林分类器
from sklearn.neighbors import KNeighborsClassifier    #s-lKNN分类器
from sklearn.svm import SVC                           #s-l支持向量机分类器
from sklearn import metrics                           #计算评价指标
from sklearn import cross_validation                  #导入交叉验证模块
import time                 

#加载语料库
print('加载语料库...')
corpus_root_reviews=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\merge\6000"
corpus_root_neg=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\neg_pre6000"
corpus_root_pos=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\pos_pre6000"

reviews=PlaintextCorpusReader(corpus_root_reviews,'.*')
neg=PlaintextCorpusReader(corpus_root_neg,'.*')
pos=PlaintextCorpusReader(corpus_root_pos,'.*')

documents_neg =[(list(neg.words(fileid)),0)   
            for fileid in neg.fileids()]
documents_pos =[(list(pos.words(fileid)),1)
            for fileid in pos.fileids()]

documents_neg.extend(documents_pos)
documents=documents_neg  
random.shuffle(documents)   #随机打乱

all_words = nltk.FreqDist(w for w in reviews.words()) #词频信息
word_features=[word for (word, freq) in all_words.most_common(3000)]  #特征词（出现频率最高的若干词）

# 特征提取函数              
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

## 特征提取、语料转化为 one-hot
print('转化 one-hot ...')
featuresets = [(list(document_features(d).values()), c) for (d,c) in documents] 
                                      
#======================================  使用scikit-learn模块分类 ====================================
x=[d for (d,c) in featuresets] #可供scikit-learn训练的输入
y=[c for (d,c) in featuresets] #可供scikit-learn训练的输出（标签）

   
clf0 = KNeighborsClassifier()
clf1 = BernoulliNB()
clf2 = SVC() 

# 交叉验证 Acc
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


'''

# 交叉验证 F
print('KNN交叉验证...')  
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='f1_macro')  
print('NB交叉验证...')  
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='f1_macro')
print('SVM交叉验证...')  
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='f1_macro')
print ( 'Fscore:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean())) 


# 交叉验证 Precision
print('KNN交叉验证...')  
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='precision')  
print('NB交叉验证...')  
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='precision')
print('SVM交叉验证...')  
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='precision')
print('RF交叉验证...')  
scores3 = cross_validation.cross_val_score(clf3, x , y , cv=3, scoring='precision')
print ( 'Precision:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n RF:{3:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean(),scores3.mean())) 


# 交叉验证 Recall
print('KNN交叉验证...')  
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='recall')  
print('NB交叉验证...')  
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='recall')
print('SVM交叉验证...')  
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='recall')
print('RF交叉验证...')  
scores3 = cross_validation.cross_val_score(clf3, x , y , cv=3, scoring='recall')
print ( 'Recall:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n RF:{3:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean(),scores3.mean()))   
  
'''    