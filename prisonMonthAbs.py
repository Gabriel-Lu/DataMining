'''
打开“result.txt"，每个事件之间按照EmptyOrEnd分开的
用正则表达式
若匹配到”有期徒刑y个月“,则向该一维数组中插入y个月;
否则，count++
至此，有期徒刑的一维数组完全形成 并打印出来
；;;缓刑的年限同理
'''
import csv
import re
import numpy as np

prisonMCount = np.zeros(652)

pattern1 = re.compile(r'有期徒刑(.*?)个月')  # 用于匹配 有期徒刑y个月

caseC = 0  # 用于案例计数
CNNum = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, }  # 此字典用于把汉字转化为阿拉伯数字
keyNum = CNNum.keys()
# 要读的文件是result.txt
for line in open("result.txt", "r"):
    print(line,end='')#",end=''"的作用是忽略掉换行符
    result1 = pattern1.findall(str(line))

    if (line[0] == 'E'):
        caseC = caseC + 1  # caseC用于计数，表示当前计数到第几个案件了
    elif(result1): # 模式匹配，把“有期徒刑xx月”中的“xx”提取出来
        # 如果是“有期徒刑x个月”而不是"缓刑y年“or"EmptyOrEnd"
        #print(result1[0])#若直接写print(result)将会打印一个列表;由于有期徒刑年限很少超过十个月，这里我们只取？年的第一个字符作为年份
        #print(str(result1[0])[-1])
        # RISKRISKRISK:若有期徒刑的年份超过了十年，那么结果就错了，健壮性有待增强
        # 把汉字“三”转化成阿拉伯数字‘3’
        if (str(result1[0])[-1] in keyNum):
            print(CNNum.get(str(result1[0])[-1]))
            prisonMCount[caseC] = CNNum.get(str(result1[0])[-1])

print("Prison Month Metrix is:\n")
print(str(prisonMCount).replace('.',','))


