#使用Matlotlib绘制决策树
#代码来自《机器学习实战》一书

import matplotlib.pyplot as plt

#设置文本框和箭头格式
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
plt.rcParams['font.family']=['STFangsong']  #使用系统字体，让画出来的图支持 中文

#画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt,\
    xycoords = "axes fraction", xytext = centerPt, textcoords = 'axes fraction',\
    va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)

#test
# def createPlot():
#     fig = plt.figure(1, facecolor = 'white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon = False)
#     plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

#获取决策树的叶子节点数
def getNumLeafs(myTree):
    leafNumber = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if(type(secondDict[key]).__name__ == 'dict'):
            leafNumber = leafNumber + getNumLeafs(secondDict[key])
        else:
            leafNumber += 1
    return leafNumber

#获取决策树的高度（递归）
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

#在父子节点添加信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

#画树
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

#画布初始化
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

#产生数据
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'} } } },
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                   {'纹理': {'稍糊': {'触感': {'硬滑': '否', '软粘': '是'}}, '清晰': {'根蒂': {'蜷缩': '是', '稍蜷': {'色泽': {'青绿': '是', '乌黑': {'触感': {'硬滑': '是', '软粘': '否'}}, '浅白': '是'}}, '硬挺': '否'}}, '模糊': '否'}}
                  ]
    return listOfTrees[i]

def main():
    #createPlot()

    myTree = retrieveTree(2)
    print(getTreeDepth(myTree))
    print(getNumLeafs(myTree))
    createPlot(myTree)

if __name__ == "__main__":
    main()