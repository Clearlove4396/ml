#数据集来自《机器学习实战》
#预测一匹病马是否能被治愈
#原始数据集来自http://archive.ics.uci.edu/ml/datasets/Horse+Colic
#其中：最后一行为类别，其他的为特征

import numpy as np
from matplotlib import pyplot as plt

#画出sigmoid函数
def plotSigmoid():
    x = np.arange(-4, 4, 0.1)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y, '-')
    plt.xlim(-4, 4)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('Sigmoid(x)')
    plt.show()

def readData(filename):
    dataSet = []
    label = []
    ifile = open(filename)
    lines = ifile.readlines()
    for line in lines:
        lineList = line.strip().split('\t')
        data = [float(lineList[i]) for i in range(len(lineList)-1)]
        data.append(1)
        dataSet.append(data)
        label.append(float(lineList[-1]))
    ifile.close()
    return dataSet, label

# f(x) = sigmoid(x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#训练
#iteration默认为100，即100次迭代
def train(trainDataSet, trainLabel, iteration = 100):
    #转化为矩阵
    trainDataMat = np.mat(trainDataSet)
    #trainLabelMat = np.mat(trainLabel).transpose()
    m, n = np.shape(trainDataMat)
    w = np.ones((n, 1))

    #开始迭代
    for i in range(iteration):
        indexList = list(range(m))
        for j in range(m):
            alpha = 4 / (1.0 + i + j) * 0.01  #学习速率动态变化
            index = int(np.random.uniform(0, len(indexList)))  #从indexList中取出一个index用于训练
            z = sigmoid(np.sum(trainDataMat[index] * w))
            dw = (trainLabel[index] - z) * trainDataMat[index].transpose()
            w = w + alpha * dw
            del(indexList[index])
    return w

def test(testDataSet, testLabel, w):
    testDataMat = np.mat(testDataSet)
    m, n = np.shape(testDataMat)
    rightCount = 0
    for i in range(m):
        z = sigmoid(np.sum(testDataMat[i] * w))
        if(z > 0.5 and testLabel[i] == 1 or z < 0.5 and testLabel[i] == 0):
            rightCount += 1
            print("right! you Label %f" % z)
        else:
            print("error! real Label: %d, you Label %f" % (testLabel[i], z))
    return float(rightCount) / m

def main():
    #读取训练数据
    trainDataSet, trainLabel = readData("horseColicTraining.txt")
    #训练
    w = train(trainDataSet, trainLabel, iteration = 500)   # w = (w1, w2, w3, ....wn, b)
    #读取测试数据
    testDataSet, testLabel = readData("horseColicTest.txt")
    #测试
    accuracyRate = test(testDataSet, testLabel, w)  # 70%的正确率
    print("total: %d, accuracyRate: %f" % (len(testDataSet), accuracyRate))

if __name__ == '__main__':
    main()
    #plotSigmoid()