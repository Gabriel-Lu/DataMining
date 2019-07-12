# -*- coding: utf-8 -*-
import os
# ltp模型目录的路径
# LTP语言技术平台是哈工大的NLP工具库，提供功能：中文分词，词性标注，命名实体识别，依存句法分析，语义角色标注
LTP_DATA_DIR = '/home/zhang/ltp_data_v3.4.0'  #使用pyltp前，要先下载LTP模型文件
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

from pyltp import Postagger #Postagger用于词性标注
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

f1 = open("seg_list.txt","r",encoding="utf-8")
words = f1.read().splitlines()  # 去掉回车
print(len(words)) # 打印words的长度
# 词性标注，KEY，生成键值对
postags = postagger.postag(words[:110]) # 疑问：为什么不分词，直接对整个文段进行词性标注？为什么只取了前111个（长度/个数“进行标注？
post_list=list(postags)
print(post_list[:100])

postagger.release()  # 释放模型
f1.close()
