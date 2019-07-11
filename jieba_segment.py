'''
作用：把test.txt中的文字进行分词，把结果存储到words_list_jieba.txt中
'''

import jieba
from jieba import analyse
''' 
with ... as NICE!
with语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常，都会执行必要的“清理”操作，释放资源
'''
with open("test.txt", "r", encoding='utf-8') as f1:
    text = f1.read() #读文件所有内容，返回一个字符串
    # 这里我们把test.txt进行分词，并把分词结果以generator的数据格式存到seg_list中
    seg_list = jieba.cut(text) #cut函数返回一个可以迭代的generator
    f2 = open("words_list_jieba.txt", "a", encoding='utf-8')#以a方式打开：在文件尾部添加内容
    #下面这个for把test.txt文本中的内容分词的结果存入到words_list_jieba.txt中
    for word in seg_list:
        f2.write(word + " ")#各个词语之间用空格分开
    f2.close()
