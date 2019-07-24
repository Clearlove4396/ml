import numpy as np
from os import listdir

def img2vector(filename):
    returnVector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(line[j])
    return returnVector

##计算距离
##排序
##选择
def knnCore(trainMat, trainLabel, testData, k):
    m = trainMat.shape[0]
    ##计算欧式距离
    diffMat = np.tile(testData, (m, 1)) - trainMat   #np.tile()  将testData的维度拓展和trainMat的维度想同
    squareDiffMat = diffMat ** 2
    squareDistance = squareDiffMat.sum(axis = 1)  #axis = 1 按行相加  axis = 0 按列相加
    distances = squareDistance ** 0.5

    ##排序
    sortedDistance = distances.argsort()  ##argsort()返回的是数组值从小到大的  索引值！！

    ##选择
    classCount = {}
    for i in range(k):
        votelLabel = trainLabel[sortedDistance[i]]
        classCount[votelLabel] = classCount.get(votelLabel, 0) + 1  #.get(self, key, default)
    #  sorted(dic,value,reverse)
    #  True--降序，False--升序（默认）
    #  key = lambda asd:asd[0]--键   key = lambda asd:asd[1]--键值
    #  对字典排序暂时不会。。。
    sortedClassCount = sorted(classCount.items(), key = lambda asd:asd[1], reverse = True)
    return sortedClassCount[0][0]

def main():
    #trainData = img2vector('data/trainingDigits/0_0.txt')

    ##读取训练数据
    trainLabel = []
    trainFileList = listdir('data/trainingDigits')
    m = len(trainFileList)
    trainMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        label = int(fileStr.split('_')[0])
        trainLabel.append(label)
        trainMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)

    ##读取测试数据，每读取一条测试数据就开始计算“距离”
    k = 3
    testFileList = listdir('data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        testLabel = int(fileStr.split('_')[0])
        testData = img2vector('data/testDigits/%s' % fileNameStr)

        knnResult = knnCore(trainMat, trainLabel, testData, k)
        print("knn result: %d, the real result: %d" % (knnResult, testLabel))
        if(knnResult != testLabel):
            errorCount += 1
    print("error number：%d，test total number: %d" % (int(errorCount), mTest))
    print("error rate：%f" % (errorCount / float(mTest)))

if __name__ == "__main__":
    main()