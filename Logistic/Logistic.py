# logistic
# 数据集来自《机器学习实战》

from matplotlib import pyplot as plt
import numpy as np

#读取数据
def readData():
    filename = "testSet.txt"
    ifile = open(filename)
    lines = ifile.readlines()
    trainDataSet = []
    trainLabel = []
    for line in lines:
        line = line.split('\n')[0]
        lineList = line.split('\t')
        lineList = [float(x) for x in lineList]
        trainDataSet.append(lineList[:-1])
        trainDataSet[-1].append(1)
        trainLabel.append(lineList[-1])
    ifile.close()
    return trainDataSet, trainLabel

# 绘制数据的散点图
def plotScatter(trainDataSet, trainLabel):
    labelList = [0, 1]
    m = len(trainLabel)
    for label in labelList:
        x1 = []
        x2 = []
        for i in range(m):
            if(trainLabel[i] == label):
                x1.append(trainDataSet[i][0])
                x2.append(trainDataSet[i][1])
        x1 = np.array(x1)
        x2 = np.array(x2)
        if(label == labelList[0]):
            plt.plot(x1, x2, 'o')
        else:
            plt.plot(x1, x2, '*')
    plt.xlim(-4, 4)
    plt.ylim(-5, 20)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#绘制回归直线
def plotBestFitLine(trainDataSet, trainLabel, w):
    labelList = [0, 1]
    m = len(trainLabel)
    for label in labelList:
        x1 = []
        x2 = []
        for i in range(m):
            if(trainLabel[i] == label):
                x1.append(trainDataSet[i][0])
                x2.append(trainDataSet[i][1])
        x1 = np.array(x1)
        x2 = np.array(x2)
        if(label == labelList[0]):
            plt.plot(x1, x2, 'o')
        else:
            plt.plot(x1, x2, '*')
    #画直线
    x = np.arange(-3.0, 3.0, 0.1)
    w = np.array(w)
    y = -(w[0][0] * x + w[0][2]) / w[0][1]
    plt.plot(x, y, '-')
    plt.xlim(-4, 4)
    plt.ylim(-5, 20)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#运用梯度上升法 - 训练数据
def train(trainDataSet, trainLabel):
    trainDataSet = np.mat(trainDataSet)
    trainLabel = np.mat(trainLabel)
    learnRate = 0.01
    w = np.mat(np.random.rand(1, np.shape(trainDataSet)[1]))  #权重初始化
    iteration = 100  #100次迭代
    for i in range(iteration):
        print(w)
        hxi = sigmoid(w * trainDataSet.transpose())
        dw = (trainLabel - hxi) * trainDataSet
        w = w + learnRate * dw
    return w

def main():
    #读取数据
    trainDataSet, trainLabel = readData()
    #绘制数据散点图
    #plotScatter(trainDataSet, trainLabel)
    #训练数据
    w = train(trainDataSet, trainLabel)  #w = (w1, w2, ...wn, b)
    #绘制直线
    plotBestFitLine(trainDataSet, trainLabel, w)

if __name__ == "__main__":
    main()