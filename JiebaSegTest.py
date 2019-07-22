#-*- coding: utf-8 -*-
import sys
sys.path.append("../")
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode:","/ ".join(seg_list))#全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode:","/ ".join(seg_list)) #默认模式


