from os import listdir
import numpy as np

#读取数据
def readData(fileFolder):
    fileList = listdir(fileFolder)
    trainDataSet = []
    trainLabel = []
    for fileName in fileList:
        #label
        label = [0 for i in range(10)]
        labelDigit = int(fileName.split('_')[0])
        label[labelDigit] = 1
        trainLabel.append(label)
        #data
        ifile = open(fileFolder + "/" + fileName)
        lines = ifile.readlines()
        dataSet = []
        for line in lines:
            line = line.split('\n')[0]
            for i in range(len(line)):
                dataSet.append(int(line[i]))
        trainDataSet.append(dataSet)

    return trainDataSet, trainLabel

def changeFrom(trainLabel, index):
    recY = []
    for label in trainLabel:
        y = [-1]
        if(label[index] == 1):
            y[0] = 1
        recY.append(y)
    return recY

#训练模型
def train_perceptron(trainDataSet, y):
    m = len(trainDataSet)  #  数据量
    n = len(trainDataSet[0]) #  特征数量
    learn_rate = 1
    #初始化模型参数
    w = np.zeros((1, n))
    b = 0
    #开始训练
    hasErrorData = True
    while(hasErrorData == True):
        hasErrorData = False
        for i in range(m):
            data = np.array(trainDataSet[i])
            #错误数据
            if(((w.dot(data.T) + b) * y[i][0]) <= 0):
                hasErrorData = True
                w = w + learn_rate * y[i][0] * data
                b = b + learn_rate * y[i][0]
                # print(w)
                # print(b)
                # print()
    return w,b

def printErrorDigit(data):
    for i in range(1, len(data) + 1):
        print(data[i-1], end = '')
        if(i % 32 == 0):
            print()

# 测试
def test_perceptron(testDataSet, testY, w, b):
    m = len(testDataSet)
    errorCount = 0
    for i in range(m):
        data = np.array(testDataSet[i])
        if((w.dot(data.T) + b) * testY[i][0] > 0):
            print("right: %d, calc: %d" % (testY[i][0], w.dot(data.T) + b))
        else:
            errorCount += 1
            print(printErrorDigit(testDataSet[i]))
            input("press any key to continue:")
    return float(errorCount) / m

def main():

    # 李航例题
    # 一致
    # trainDataSet = [[3, 3], [4, 3], [1, 1]]
    # trainLabel = [[1], [1], [-1]]
    # w, b = train_perceptron(trainDataSet, trainLabel)
    # print(w)
    # print(b)
    # print()

    # mnist
    trainDataSet, trainLabel = readData("data/trainingDigits")
    y = changeFrom(trainLabel, 0)    # 数字0作为正(y = 1)， 其他数字作为负（y = -1)
    w, b = train_perceptron(trainDataSet, y)

    testDataSet, testLabel = readData("data/testDigits")
    testY = changeFrom(testLabel, 0)
    errorRate = test_perceptron(testDataSet, testY, w, b)
    print("total: %d, errorRate: %f" % (len(testDataSet), errorRate))

if __name__ == "__main__":
    main()