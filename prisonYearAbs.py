'''
打开“result.txt"，每个事件之间按照EmptyOrEnd分开的
用正则表达式
若匹配到”有期徒刑x年“,则向该一维数组中插入x*12个月;
否则，count++
至此，有期徒刑的一维数组完全形成 存储到prisonYear.csv
；;;缓刑的年限同理
'''
import csv
import re
import numpy as np

pattern = re.compile(r'有期徒刑(.*?)年')#用于模式匹配
caseC=0#用于案例计数
CNNum={'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,}#此字典用于把汉字转化为阿拉伯数字
keyNum=CNNum.keys()
#要读的文件是result.txt
for line in open("result.txt", "r"):
    print(line,end='')#",end=''"的作用是忽略掉换行符
    if (line[0] == 'E'):
        caseC = caseC + 1#caseC用于计数，表示当前计数到第几个案件了
    else:
        result = pattern.findall(str(line))#模式匹配，把“有期徒刑三年”中的“三”提取出来
        if(result):
            print(result[0])#若直接写print(result)将会打印一个列表;由于有期徒刑年限很少超过十年，这里我们只取？年的第一个字符作为年份
            #RISKRISKRISK:若有期徒刑的年份超过了十年，那么结果就错了，健壮性有待增强

            #把汉字“三”转化成阿拉伯数字‘3’
            #7.23做这个，参考AbstractTime.py中标成蓝色的代码

        '''
        if (result):
            print('------result------' + str(caseC))
            # result=list(map(int,result))
            print(result[0])
            # prisonYCount[caseC] = result[0]
            writer.writerow(result[0])
        '''


'''
#要写入文件prisonYear.csv
prisonFile=open("prisonYear.csv","a")
writer=csv.writer(prisonFile)


prisonYCount = np.zeros(652)
caseC=0#第几个案例
for line in reader:
    print('line')
    print(line)#test
'''

'''
        for item in line:
        print('item')
        print(item)#Testing
        #var=var+item
        var=item.split(' ')
        print('var')
        print(var)
        for item in range(len(var)):
            print(var[item])#Testing
            if(var[item][0]=='E'):
                caseC=caseC+1
                writer.writerow('end or none')
            else:
                


print(prisonYCount)



prisonFile.close()
csvFile.close()
    '''



