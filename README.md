# DataMining
## 输入raw500.csv
- 内容是{词语，词性}格式的
## 按照词性对词语进行分类
> Delete0.py
- 词性中B，I，E代表begin,mid,end 。N和P代表negative 和positive
- 按照积极positive和消极negative和result对事件进行分类，分成3类
> generateResult.py 把结果result中的BR，IR，ER连接起来，写入到result.txt文件
## Bool型事件的抽取
> NegMetrix.py
- 建立一个大的一维数组，每个数值代表一个案例在该要素的bol值。若词语中出现了那个词、则为1.否则为0
## 数值型事件的抽取
- 由于预测缓刑/有期徒刑时间以月份为单位，而文本中时间的表示形式是：x年y个月。所以先抽取出年份、再抽取出月份，然后相加
> prisonYearAbs.py 年份抽取到一维数组
> prisonMonthAbs.py 把月份抽取到一维数组。
- prisonMonthAbs其实抽取出来的是x年y.我们只取字符串最后一个字符y即可

## 汉字数字表示转化成阿拉伯数字表示： 字典

