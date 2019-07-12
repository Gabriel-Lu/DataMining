#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Author:Zhang Shiwei
# Function: 把文件里面的单个字全部抽取出来，放到f2文件中
with open("dict","r",encoding="utf-8") as f1:
    words = f1.read().splitlines() # 去掉换行
    print(len(words))
 #   print(words[:10])
    name_list = list(set(words)) #set()用于创建集合。这里把word中的所有单字都挑出来，创建了一个集合
    print(len(name_list)) # 打印words里面有多少个不同的字
    f2 = open("name","a",encoding='utf-8')
    # 把word里面的字都打印出来，每行只打印一个字
    for i in name_list:
        f2.write(i+"\n")
