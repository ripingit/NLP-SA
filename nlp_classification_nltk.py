import nltk
import random
from nltk.classify import SklearnClassifier           #nltk中使用scikit-learn
from sklearn.naive_bayes import BernoulliNB           #s-l贝叶斯分类器
from sklearn.ensemble import RandomForestClassifier   #s-l随机森林分类器
from sklearn.neighbors import KNeighborsClassifier    #s-lKNN分类器
from sklearn.svm import SVC                           #s-l支持向量机分类器
from sklearn import metrics                           #计算评价指标
from sklearn.cross_validation import train_test_split #划分训练集和测试集 

#加载自己的语料库
from nltk.corpus import PlaintextCorpusReader
corpus_root_reviews=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000"
corpus_root_neg=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\neg_pre"
corpus_root_pos=r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\pos_pre"

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

'''
======================================================================================================
======================================================================================================
======================================================================================================
======================================================================================================
======================================================================================================
'''

'''
all_words = nltk.FreqDist(w for w in reviews.words()) #词频信息
word_features=[word for (word, freq) in all_words.most_common(3000)]  #特征词

#特征提取函数              
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents] 
#featuresets0 = [(list(d.values()),c) for (d,c) in featuresets] #
#featuresets = [(list(document_features(d).values()), c) for (d,c) in documents] #语料转化为one-hot向量


                
                
#======================================= 使用nitk模块 分类 ==========================================     
          
#train_set,test_set=featuresets[1000:],featuresets[:1000]#训练集、测试集
#classifier0=nltk.NaiveBayesClassifier.train(train_set)
#classifier1=SklearnClassifier(RandomForestClassifier()).train(train_set)
#classifier2=SklearnClassifier(BernoulliNB()).train(train_set)
#classifier3=SklearnClassifier(SVC()).train(train_set)

#print (nltk.classify.accuracy(classifier0,test_set))
#print (nltk.classify.accuracy(classifier1,test_set))
#print (nltk.classify.accuracy(classifier2,test_set))
#print (nltk.classify.accuracy(classifier3,test_set))

#哪些特征是分类器发现最有信息量的
#classifier0.show_most_informative_features(20)#只有nltk中的分类器可用
    