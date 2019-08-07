#   使用朴素贝叶斯算法进行邮件分类
#   非垃圾邮件：2  email/ham
#   垃圾邮件:   -2  email/spam

import re
from os import listdir
import random

def getVocabularyList():
    vocabularySet = set([])
    fileList = listdir('email/ham')  # 正常邮件
    m = len(fileList)
    fileList.extend(listdir('email/spam'))  #非正常邮件
    regEx = re.compile('\\W*')
    for filename in fileList:
        if(m != 0):
            m -= 1
            ifile = open('email/ham' + '/' + filename)
        else:
            ifile = open('email/spam' + '/' + filename)
        lines = ifile.readlines()
        s = ''
        for line in lines:
            s = s + " " + line.split('\n')[0]
        listOfTokens = regEx.split(s)
        vocabularySet = vocabularySet | set(listOfTokens)
        ifile.close()
    return list(vocabularySet)

def readData(vocabularList):
    dataSet = []
    label = []
    dirs = ['email/ham', 'email/spam']
    regEx = re.compile('\\W*')
    for dir in dirs:
        fileList = listdir(dir)
        for filename in fileList:
            data = [0] * len(vocabularList)
            ifile = open(dir + '/' + filename)
            lines = ifile.readlines()
            for line in lines:
                line = line.split('\n')[0]
                listOfTokens = regEx.split(line)
                for token in listOfTokens:
                    index = vocabularList.index(token)
                    data[index] = 1
            dataSet.append(data)
            if(dir == dirs[0]): #非垃圾邮件
                label.append(2)
            else:
                label.append(-2) #垃圾邮件
    #使用相同的规则打乱两个列表
    zipTmp = list(zip(dataSet, label))
    random.shuffle(zipTmp)
    dataSet[:], label[:] = zip(*zipTmp)
    # 拆分成train 和 test数据集
    trainM = int(len(label) * 0.7)  # 前70%用于训练
    trainDataSet = dataSet[:trainM][:]
    trainLabel = label[:trainM]
    testDataSet = dataSet[trainM:][:]
    testLabel = label[trainM:]

    return trainDataSet, trainLabel, testDataSet, testLabel

#训练
# -- 获得先验概率和条件概率
def train(trainDataSet, trainLabel, vocabularyList):
    classSet = set(trainLabel)  # 标签的取值范围
    m = len(trainLabel)  #训练样本数
    k = len(classSet) #类别个数
    lambd = 1   #拉普拉斯平滑系数
    prob_ck = {}
    #计算p(Y = ck)
    for ck in classSet:
        prob_ck[ck] = float(trainLabel.count(ck) + lambd) / (m + k * lambd)
    # 计算条件概率  p(Xi = xi | Y = ck)
    #tmp = [0] * len(vocabularyList)
    prob_condition = {}
    for ck in classSet:
        prob_condition[ck] = {}
    for key in prob_condition.keys():
        sumCk = trainLabel.count(key)
        for j in range(len(trainDataSet[0])):
            #XjSet = set(x[j] for x in trainDataSet)
            XjSet = set([0, 1])  #不能用上面的一句，因为一定要包含所有的！
            Sj = len(XjSet)
            prob_condition[key][j] = {}
            for aji in XjSet:
                res = []
                for i in range(m):
                    if(trainDataSet[i][j] == aji):
                        res.append(i)
                countList = [i for i in res if trainLabel[i] == key]
                sumXji = len(countList)
                prob_condition[key][j][aji] = (float)(sumXji + lambd) / (sumCk + Sj * lambd)
    return prob_ck, prob_condition

#测试
def test(testDataSet, testLabel, vocabularyList, prob_ck, prob_condition):
    m = len(testDataSet)
    n = len(testDataSet[0])
    errorCount = 0
    for i in range(m):
        maxProb = -10
        maxLabel = -1
        for key in prob_ck.keys():
            currentProb = prob_ck[key]
            ckDict = prob_condition[key]
            for j in range(n):
                #print(testDataSet[i][j])
                currentProb = currentProb * ckDict[j][testDataSet[i][j]]
            if(currentProb > maxProb):
                maxLabel = key
                maxProb = currentProb
        if(maxLabel == testLabel[i]):
            print("right! %d" % maxLabel)
        else:
            print("error! realLabel: %d  predictLabel %d: " % (testLabel[i], maxLabel))
            errorCount += 1
    print("total: %d  errorRate: %f" % (m, float(errorCount) / m))

def main():
    vocabularList = getVocabularyList()
    #print(vocabularList)
    trainDataSet, trainLabel, testDataSet, testLabel = readData(vocabularList)
    #训练 - 获得各种概率
    prob_ck, prob_condition = train(trainDataSet, trainLabel, vocabularList)
#    print(prob_ck)
#    print(prob_condition)

    #测试
    test(testDataSet, testLabel, vocabularList, prob_ck, prob_condition)

if __name__ == "__main__":
    main()