# 将neg或pos评论合并在一个文件内
import glob
# pos
f0=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\pos.txt",'w',encoding='utf-8')
for file in glob.glob(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\pos_pre\*.txt"):
    with open(file,"r+",encoding= 'utf-8') as f1:
        lines=f1.readlines()
        lines1=''.join(lines)
        f0.write(lines1+'\n')
f0.close()

# neg
f0=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\neg.txt",'w',encoding='utf-8')
for file in glob.glob(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\neg_pre\*.txt"):
    with open(file,"r+",encoding= 'utf-8') as f1:
        lines=f1.readlines()
        lines1=''.join(lines)
        f0.write(lines1+'\n')
f0.close()

# 6000
f0=open(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\2000.txt",'w',encoding='utf-8')
for file in glob.glob(r"E:\Strange\SRTP.11\NLP\data\ChnSentiCorp_htl_ba_2000\*.txt"):
    with open(file,"r+",encoding= 'utf-8') as f1:
        lines=f1.readlines()
        lines1=''.join(lines)
        f0.write(lines1+'\n')
f0.close()

# 需要再去除一下几个衔接处的空行

        
                           