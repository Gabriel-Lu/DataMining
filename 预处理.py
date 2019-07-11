# function:把“原始数据.txt”中的标点符号，括号及括号内的注释，本院认为|违反交通。。。。去掉之后，放入到“预处理后数据.txt"

import re
punc = "，。、:；（）"
f1 = open("原始数据.txt", "r", encoding='utf-8')
f2 = open("预处理后数据.txt", "w", encoding='utf-8')
for line in f1.readlines():
    line1 = re.sub(u"\\（.*?\\）","", line)       #去除括号内注释
    line2 = re.sub("[%s]+" % punc, "", line1)   # 去除标点
    #re.sub()函数用于替换字符串中的匹配项，下面的语句把“本院认为|违反交通。。。。”替换成“”，即删除掉了
    line3 = re.sub("本院认为|违反交通运输管理法规|违反道路交通管理法规|驾驶机动车辆|因而|违反道路交通运输管理法规|违反交通运输管理法规|缓刑考.*?计算|刑期.*?止|依照|《.*?》|第.*?条|第.*?款|的|了|其|另|已|且", "", line2)
    f2.write(line3)#把字符串line3写入到文件f2
f1.close()
f2.close()
