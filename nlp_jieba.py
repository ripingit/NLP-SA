# 格式预处理、去停止词分词

import re
import glob
import jieba 

punctuations = ['的','着','之','了','也'] # 停止词

i=0;
for file in glob.glob(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\pos\*.txt"):
    with open(file,"r+",encoding= 'utf-8') as f1:
        # f1=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\neg\neg.0.txt",encoding='utf-8')
        # f1=open(r"E:\Strange\SRTP.11\NLP\data\wiki\wiki_cn.txt",encoding='utf-8',errors='ignore') # wiki
        lines=f1.readlines() # list
        lines=''.join(lines) # str
        lines=lines.replace('\n', '') # 删除空行 str
        lines=''.join(re.findall(u'[\u4e00-\u9fa5]+', lines)) # 去除非汉字 str
        seg_list = jieba.cut(lines) # 结巴分词 
        seg_list=[word for word in seg_list if not word in punctuations] # 去除停止词，list
        seg_list=' '.join(seg_list) # str
        # print(seg_list) #显示分词结果
        f2=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\pos_pre\%d.txt"%i,'w',encoding='utf-8') # 结果文件夹
        f2.write(seg_list) # 将结果写入文件
        f2.close()
        i=i+1;
        
        
i=0;
for file in glob.glob(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\neg\*.txt"):
    with open(file,"r+",encoding= 'utf-8') as f1:
        # f1=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_6000\neg\neg.0.txt",encoding='utf-8')
        # f1=open(r"E:\Strange\SRTP.11\NLP\data\wiki\wiki_cn.txt",encoding='utf-8',errors='ignore') # wiki
        lines=f1.readlines() # list
        lines=''.join(lines) # str
        lines=lines.replace('\n', '') # 删除空行 str
        lines=''.join(re.findall(u'[\u4e00-\u9fa5]+', lines)) # 去除非汉字 str
        seg_list = jieba.cut(lines) # 结巴分词 
        seg_list=[word for word in seg_list if not word in punctuations] # 去除停止词，list
        seg_list=' '.join(seg_list) # str
        # print(seg_list) #显示分词结果
        f2=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\neg_pre\%d.txt"%i,'w',encoding='utf-8') # 结果文件夹
        f2.write(seg_list) # 将结果写入文件
        f2.close()
        i=i+1;        
