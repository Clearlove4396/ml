#利用决策树算法，对mnist数据集进行测试

from os import listdir
import numpy as np

#读取数据
# fileDir ： 数据文件所在的文件夹目录
def readDataSet(fileDir):
    fileList = listdir(fileDir)
    m = len(fileList)
 #   dataSet = np.zeros((m, 1025))
    dataSet = []
    index = 0
    for fileName in fileList:
        fr = open(fileDir + "/" + fileName)
        data = []
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                data.append(int(line[j]))
        dataSet.append(data)
        dataLabel = int((fileName.split('.')[0]).split('_')[0])
        dataSet[index][-1] = str(dataLabel)
        index += 1
    featureNamesSet = []
    for i in range(len(dataSet[0]) - 1):
        col = [x[i] for x in dataSet]
        colSet = set(col)
        featureNamesSet.append(list(colSet))
    return dataSet, featureNamesSet

#计算熵
def calcEntropy(dataSet):
    mD = len(dataSet)
    dataLabelList = [x[-1] for x in dataSet]
    dataLabelSet = set(dataLabelList)
    ent = 0
    for label in dataLabelSet:
        mDv = dataLabelList.count(label)
        prop = float(mDv) / mD
        ent = ent - prop * np.math.log(prop, 2)

    return ent

# # 拆分数据集
# # index - 要拆分的特征的下标
# # feature - 要拆分的特征
# # 返回值 - dataSet中index所在特征为feature，且去掉index一列的集合
def splitDataSet(dataSet, index, feature):
    splitedDataSet = []
    for data in dataSet:
        if(data[index] == feature):
            reduceFeatureVec = data[: index]
            reduceFeatureVec.extend(data[index + 1 :])
            splitedDataSet.append(reduceFeatureVec)
    return splitedDataSet

#根据信息增益 - 选择最好的特征
# 返回值 - 最好的特征的下标
def chooseBestFeature(dataSet):
    entD = calcEntropy(dataSet)
    mD = len(dataSet)
    featureNumber = len(dataSet[0]) - 1
    maxGain = -100
    maxIndex = -1
    for i in range(featureNumber):
        entDCopy = entD
        featureI = [x[i] for x in dataSet]
        featureSet = set(featureI)
        for feature in featureSet:
            splitedDataSet = splitDataSet(dataSet, i, feature)  # 拆分数据集
            mDv = len(splitedDataSet)
            entDCopy = entDCopy - float(mDv) / mD * calcEntropy(splitedDataSet)
        if(maxIndex == -1):
            maxGain = entDCopy
            maxIndex = i
        elif(maxGain < entDCopy):
            maxGain = entDCopy
            maxIndex = i
    return maxIndex

# 寻找最多的，作为标签
def mainLabel(labelList):
    labelRec = labelList[0]
    maxLabelCount = -1
    labelSet = set(labelList)
    for label in labelSet:
        if(labelList.count(label) > maxLabelCount):
            maxLabelCount = labelList.count(label)
            labelRec = label
    return labelRec

# 生成树
# def createDecisionTree(dataSet, featureNames):
#     labelList = [x[-1] for x in dataSet]
#     if(len(dataSet[0]) == 1): #没有可划分的属性了
#         return mainLabel(labelList)  #选出最多的label作为该数据集的标签
#     elif(labelList.count(labelList[0]) == len(labelList)): # 全部都属于同一个Label
#         return labelList[0]
#
#     bestFeatureIndex = chooseBestFeature(dataSet)
#     bestFeatureName = featureNames.pop(bestFeatureIndex)
#     myTree = {bestFeatureName: {}}
#     featureList = [x[bestFeatureIndex] for x in dataSet]
#     featureSet = set(featureList)
#     for feature in featureSet:
#         featureNamesNext = featureNames[:]
#         splitedDataSet = splitDataSet(dataSet, bestFeatureIndex, feature)
#         myTree[bestFeatureName][feature] = createDecisionTree(splitedDataSet, featureNamesNext)
#     return myTree

#生成 完整的 决策树
# featureNamesSet 是featureNames取值的集合
# labelListParent 是父节点的标签列表
def createFullDecisionTree(dataSet, featureNames, featureNamesSet, labelListParent):
    labelList = [x[-1] for x in dataSet]
    if(len(dataSet) == 0):
        return mainLabel(labelListParent)
    elif(len(dataSet[0]) == 1): #没有可划分的属性了
        return mainLabel(labelList)  #选出最多的label作为该数据集的标签
    elif(labelList.count(labelList[0]) == len(labelList)): # 全部都属于同一个Label
        return labelList[0]

    bestFeatureIndex = chooseBestFeature(dataSet)
    print(bestFeatureIndex)
    bestFeatureName = featureNames.pop(bestFeatureIndex)
    myTree = {bestFeatureName: {}}
    featureList = featureNamesSet.pop(bestFeatureIndex)
    featureSet = set(featureList)
    for feature in featureSet:
        featureNamesNext = featureNames[:]
        featureNamesSetNext = featureNamesSet[:][:]
        splitedDataSet = splitDataSet(dataSet, bestFeatureIndex, feature)
        myTree[bestFeatureName][feature] = createFullDecisionTree(splitedDataSet, featureNamesNext, featureNamesSetNext, labelList)
    return myTree


#读取西瓜数据集2.0
def readWatermelonDataSet():
    ifile = open("周志华_西瓜数据集2.txt")
    featureName = ifile.readline()  #表头
    featureNames = (featureName.split(' ')[0]).split(',')
    lines = ifile.readlines()
    dataSet = []
    for line in lines:
        tmp = line.split('\n')[0]
        tmp = tmp.split(',')
        dataSet.append(tmp)
    #获取featureNamesSet
    featureNamesSet = []
    for i in range(len(dataSet[0]) - 1):
        col = [x[i] for x in dataSet]
        colSet = set(col)
        featureNamesSet.append(list(colSet))

    return dataSet, featureNames, featureNamesSet

#  把决策树保存到文件中
def saveMyTree(myTree, filename):
    s = str(myTree)
    ofile = open(filename, 'w')
    ofile.writelines(s)
    ofile.close()

#  加载决策树
def loadMyTree(filename):
    # TODO: 异常检测
    ifile = open(filename, 'r')
    s = ifile.readline()
    myTree = eval(s)   # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
                        #eval('pow(2, 3)')   8
    return myTree

#  test  全局变量法
# testLabel = -1
# def getLabel(myTree, data):
#     global testLabel  #强调他是全局变量
#     keys = myTree.keys()
#     for key in keys:
#         subTree = myTree[key]
#         index = int(key)
#         feature = int(data[index])
#         nextTree = subTree[feature]
#         if(type(nextTree).__name__ == 'dict'):
#             getLabel(nextTree, data)
#         else:
#             if(testLabel == -1):
#                 testLabel = int(nextTree)
#
# def test(myTree, testDataSet):
#     errorCount = 0
#     m = len(testDataSet)
#     for data in testDataSet:
#         global testLabel
#         testLabel = -1
#         realLabel = int(data[-1])
#         getLabel(myTree, data[:-1])
#
#         if(testLabel == realLabel):
#             print("right! label: %d" % realLabel)
#         else:
#             errorCount += 1
#             print("error! realLabel: %d  forecastLabel: %d" % (realLabel, testLabel))
#     print("total number: %d, errorRate: %f" % (m, float(errorCount) / m))


#   测试
def getLabel(myTree, data):
    keys = myTree.keys()
    for key in keys:
        subTree = myTree[key]
        index = int(key)
        feature = int(data[index])
        nextTree = subTree[feature]
        if(type(nextTree).__name__ == 'dict'):
            classLabel = getLabel(nextTree, data)
        else:
            classLabel = int(nextTree)
    return classLabel

def test(myTree, testDataSet):
    errorCount = 0
    m = len(testDataSet)
    for data in testDataSet:
        realLabel = int(data[-1])
        testLabel = getLabel(myTree, data[:-1])

        if(testLabel == realLabel):
            print("right! label: %d" % realLabel)
        else:
            errorCount += 1
            print("error! realLabel: %d  forecastLabel: %d" % (realLabel, testLabel))
    print("total number: %d, errorRate: %f" % (m, float(errorCount) / m))

def main():
    # #读取数据
    # trainingDataSet, featureNamesSet = readDataSet("data/trainingDigits")
    # featureNames = [str(x) for x in range(1024)]
    #
    # #  训练 - 得到决策树myTree
    # myTree = createFullDecisionTree(trainingDataSet, featureNames, featureNamesSet, [])
    # #  保存到文件中
    # saveMyTree(myTree, "myTree.txt")

    myTree = loadMyTree("myTree.txt")
    #读取测试数据
    testDataSet, featureNamesSet = readDataSet("data/testDigits")
    test(myTree, testDataSet)


    #  西瓜数据集
    # dataSet, featureNames, featureNamesSet = readWatermelonDataSet()
    # myTree = createFullDecisionTree(dataSet, featureNames, featureNamesSet, [])
    # saveMyTree(myTree, "myTree.txt")
    # a = loadMyTree("myTree.txt")
    # print(a)

if __name__ == "__main__":
    main()