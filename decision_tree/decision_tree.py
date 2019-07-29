import numpy as np
import operator

#计算信息熵
def calcInformationEntropy(dataSet):
    #dataSet最后一列是类别，前面是特征
    dict = {}
    m = len(dataSet)
    for i in range(m):
        #.get()函数：如果没有这个key，就返回默认值；如果有这个key，就返回这个key的value
        dict[dataSet[i][-1]] = dict.get(dataSet[i][-1], 0) + 1;
    ent = 0
    for key in dict.keys():
        p = float(dict[key]) / m
        ent = ent - (p * np.math.log(p, 2))

    return ent

#划分数据集
#dataSet:数据集
#axis: 要划分的列下标
#value: 要划分的列的值
def splitDataSet(dataSet, axis, value):
    splitedDataSet = []
    for data in dataSet:
        if(data[axis] == value):
            reduceFeatureVec = data[: axis]
            reduceFeatureVec.extend(data[axis + 1 :])
            splitedDataSet.append(reduceFeatureVec)
    return splitedDataSet

#计算信息增益,然后选择最优的特征进行划分数据集
#信息增益的计算公式：西瓜书P75
def chooseBestFeatureToSplit(dataSet):
    #计算整个集合的熵
    EntD = calcInformationEntropy(dataSet)
    mD = len(dataSet)  #行
    featureNumber = len(dataSet[0][:]) - 1 #列
    maxGain = -1000
    bestFeatureIndex = -1
    for i in range(featureNumber):
        #featureSet = set(dataSet[:][i])  #错误写法：dataSet[:][i]仍然是获取行
        featureCol = [x[i] for x in dataSet]   #取列表某列的方法！！
        featureSet = set(featureCol)
        splitedDataSet = []
        for av in featureSet:
            retDataSet = splitDataSet(dataSet, i, av)
            splitedDataSet.append(retDataSet)
        gain = EntD
        for ds in splitedDataSet:
            mDv = len(ds)
            gain = gain - (float(mDv) / mD) * calcInformationEntropy(ds)
        if(bestFeatureIndex == -1):
            maxGain = gain
            bestFeatureIndex = i
        elif(maxGain < gain):
            maxGain = gain
            bestFeatureIndex = i

    return bestFeatureIndex

#当所有的特征划分完了之后，如果仍然有叶子节点中的数据不是同一个类别，
# 则把类别最多的作为这个叶子节点的标签
def majorityCnt(classList):
    dict = {}
    for label in classList:
        dict[label] = dict.get(label, 0) + 1
    sortedDict = sorted(dict, dict.items(), key = operator.itemgetter(1), reversed = True)
    return sortedDict[0][0]

#递归构建决策树
def createTree(dataSet, labels):
    classList = [x[-1] for x in dataSet]
#    if(len(set(classList)) == 1):
#        return classList[0]
    if(classList.count(classList[0]) == len(classList)):
        return classList[0]
    elif(len(dataSet[0]) == 1):  #所有的属性全部划分完毕
        return majorityCnt(classList)
    else:
        bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
        bestFeatureLabel = labels[bestFeatureIndex]
        myTree = {bestFeatureLabel: {}}
        del(labels[bestFeatureIndex])  #使用完该属性之后，要删除
        featureList = [x[bestFeatureIndex] for x in dataSet]
        featureSet = set(featureList)
        for feature in featureSet:
            subLabels = labels[:]  #拷贝一份，防止label在递归的时候被修改  （list是传引用调用）
            tmpDataSet = splitDataSet(dataSet, bestFeatureIndex, feature)  #划分数据集
            myTree[bestFeatureLabel][feature] = createTree(tmpDataSet, subLabels)
        return myTree

#自己创造一个简单的数据集
def createTestData():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels

#读取西瓜数据集2.0
def readWatermelonDataSet():
    ifile = open("周志华_西瓜数据集2.txt")
    featureName = ifile.readline()  #表头
    labels = (featureName.split(' ')[0]).split(',')
    lines = ifile.readlines()
    dataSet = []
    for line in lines:
        tmp = line.split('\n')[0]
        tmp = tmp.split(',')
        dataSet.append(tmp)

    return dataSet, labels


def main():
    dataSet, labels = createTestData()
    #dataSet[0][-1] = "maybe"
    ent = calcInformationEntropy(dataSet)
    print(ent)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    print(bestFeature)

    tree = createTree(dataSet, labels)
    print(tree)

    melonDataSet, melonLabels = readWatermelonDataSet()
    #print(melonDataSet)
    print(melonLabels)
    melonBestFeature = chooseBestFeatureToSplit(melonDataSet)
    tree = createTree(melonDataSet, melonLabels)
    print(tree)

if __name__ == "__main__":
    main()