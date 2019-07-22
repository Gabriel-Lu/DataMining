import csv
import re

csvFile = open("prisonMonth.csv", "r")
reader = csv.reader(csvFile)
#要写入文件prisonMonth.csv
prisonFile=open("prisonMonth2.csv","a")
writer=csv.writer(prisonFile)

var=''
caseC=0#第几个案例
for line in reader:
    print(line)#test
    if(line[0]!='-'):
        pattern=re.compile(r'\d+个')
        result=pattern.findall(str(line))
        print('---result---')
        print(str(result).replace('个',''))
        writer.writerow([str(result).replace('个','')])
    else:
        writer.writerow('-')

    '''
        for item in range(len(var)):
                pattern = re.compile(r'年\d+')
                result = pattern.findall(str(var[item]))
                print('------result------')
                print(result)
                if(result):
                    writer.writerow(result)
            else:
                writer.writerow('-')
    
'''

prisonFile.close()

csvFile.close()