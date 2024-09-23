import numpy as np
import math


SIGMOID_CONST = 1


def sigmoid(val):
    temp = math.exp(val / SIGMOID_CONST)
    return temp / (1.0 + temp)

def dSigmoid(val):
    temp = math.exp(val / SIGMOID_CONST)
    return temp / ((1.0 + temp) ** 2 * SIGMOID_CONST)

def calcCost(realRes, expectedRes):
    cost = 0.0
    for i in range(len(realRes)):
        cost += (realRes[i] - expectedRes[i]) ** 2
    return cost

class NeuralNetwork:
    def __init__(self, neuronCntPerLayers:list):
        self.sigmoidV = np.vectorize(sigmoid)
        if(len(neuronCntPerLayers) < 1):
            return
        self.a = [np.ndarray(neuronCntPerLayers[0])]
        self.z = [None]
        self.w = [None]
        self.b = [None]
        self.neuronCntPerLayers = neuronCntPerLayers
        self.numOfLayers = len(neuronCntPerLayers)
        
        
        for i in range(1, self.numOfLayers):
            curCnt = neuronCntPerLayers[i]
            self.a.append(np.ndarray(curCnt))
            self.z.append(np.ndarray(curCnt))
            offset = np.ndarray((curCnt, neuronCntPerLayers[i - 1]))
            offset.fill(0.5)
            self.w.append(2 * (np.random.rand(curCnt, neuronCntPerLayers[i - 1]) - offset))
            offset = np.ndarray(curCnt)
            offset.fill(0.5)
            self.b.append(np.zeros(curCnt))
            # print(self.a[i].shape)
            # print(self.w[i].shape)
            # print(self.b[i].shape)

    
    def setTrainingData(self, data, labels):
        self.trainingData = data
        self.trainingLabels = labels
    
    def setTestData(self, data, labels):
        self.testData = data
        self.testLabels = labels

    def getResFromLabel(self, label):
        res = np.zeros(self.neuronCntPerLayers[-1])
        res[label] = 1.0
        return res
    
    def getRes(self, data):
        self.a[0] = data
        # print(self.a[0])
        # reluV = np.vectorize(self.relu)
        for layer in range(1, self.numOfLayers):
            self.z[layer] = np.matmul(self.w[layer], self.a[layer - 1]) + self.b[layer]
            self.a[layer] = self.sigmoidV(self.z[layer])
            # print("A", layer, self.a[layer])
        return self.a[-1]
    
    
    def train(self, batchSize = 10, learningRate = 1.0):
        # dAavg = [None]
        dBavg = [None]
        dWavg = [None]
        print("Train start...")
        for i in range(1, self.numOfLayers):
            curCnt = self.neuronCntPerLayers[i]
            # dAavg.append(np.zeros(curCnt))
            dWavg.append(np.zeros((curCnt, self.neuronCntPerLayers[i - 1])))
            dBavg.append(np.zeros(curCnt))

        nums = np.arange(len(self.trainingData))
        np.random.shuffle(nums)
        
        for num in range(len(self.trainingData)):
            trainSampleId = nums[num]
            sampleData = self.trainingData[trainSampleId]
            sampleLabel = self.trainingLabels[trainSampleId]
            dA = [np.ones(self.neuronCntPerLayers[0])]
            dB = [None]
            dW = [None]
            for i in range(1, self.numOfLayers):
                curCnt = self.neuronCntPerLayers[i]
                dA.append(np.ones(curCnt))
                dW.append(np.ones((curCnt, self.neuronCntPerLayers[i - 1])))
                dB.append(np.ones(curCnt))
            # if sampleLabel != 0:
            #     continue
            expectedRes = self.getResFromLabel(sampleLabel)
            # print("ExpectedRes: ", expectedRes)
            self.getRes(sampleData)
            # print("RealRes", self.a[-1])
            dC = 2 * (self.a[-1] - expectedRes)
            # print("A", self.a[-1])
            # print("Y", expectedRes)
            # print("dC", dC)
            dA[-1] = dC
            for layer in range(self.numOfLayers - 1, 0, -1):
                da = dA[layer - 1]
                db = dB[layer]
                dw = dW[layer]
                curA = self.a[layer]
                curZ = self.z[layer]
                prevA = self.a[layer - 1]
                # print("PrevA", layer, prevA)
                curW = self.w[layer]
                for j in range(self.neuronCntPerLayers[layer]):
                    db[j] *= dA[layer][j] * dSigmoid(curZ[j])
                    for k in range(self.neuronCntPerLayers[layer - 1]):
                        dw[j][k] *= dA[layer][j] * dSigmoid(curZ[j]) * prevA[k]
                        da[k] = 0
                        for i in range(self.neuronCntPerLayers[layer]):
                            da[k] += dA[layer][i] * dSigmoid(curZ[i]) * curW[i][k] 
                    # if layer == 1:
                    #     print("dw", j, dw[j])
            
            if num % batchSize == batchSize - 1: 
                for i in range(1, self.numOfLayers):
                    # self.a[i] += dA[i] / batchSize
                    # print(dWavg[i].shape)
                    # print(self.w[i].shape)
                    # print(dBavg[i].shape)
                    # print(self.b[i].shape)
                    # print(dWavg)
                    # print(dBavg)
                    # print("dbavg", i,"=", dBavg[i] / batchSize)
                    # print("dwavg", i, "=", dWavg[i] / batchSize)
                    # print("daavg", i, "=", dA[i])
                    self.w[i] -= dWavg[i] / batchSize * learningRate
                    self.b[i] -= dBavg[i] / batchSize * learningRate
                    # print("w", i,"=",  self.w[i])
                    # print("b", i, "=", self.b[i])
                    
                    curCnt = self.neuronCntPerLayers[i]
                    dWavg[i] = np.zeros((curCnt, self.neuronCntPerLayers[i - 1]))
                    dBavg[i] = np.zeros(curCnt)

            for i in range(1, self.numOfLayers):
                # dAavg[i] += dA[i]
                # print(dWavg[i].shape)
                # print(dBavg[i].shape)
                # print("db", i,"=", dB[i])
                # print("dw", i, "=", dW[i])
                # print("da", i, "=", dA[i])
                dWavg[i] += dW[i]
                dBavg[i] += dB[i]
            # print("Finished " , num, " of ",  len(self.trainingData))
    
    def test(self):
        numOfCorrectAnswers = 0
        avgCost = 0.0
        for testSampleId in range(len(self.testData)):
            sampleData = self.testData[testSampleId]
            sampleLabel = self.testLabels[testSampleId]
            expectedRes = self.getResFromLabel(sampleLabel)
            realRes = self.getRes(sampleData)
            cost = calcCost(realRes, expectedRes)
            # print(expectedRes)
            # print(realRes)
            # print("Test ", testSampleId, " has cost:", cost)
            avgCost += cost
            if np.max(realRes) == realRes[sampleLabel]:
                numOfCorrectAnswers += 1
        print("Correct answers: ", numOfCorrectAnswers / len(self.testData) * 100, "%")
        return avgCost / len(self.testData)

    def calcCost(realRes: np.ndarray, num):
        expectedRes = np.zeros(len(realRes))
        expectedRes[num] = 1.0
        res = 0.0
        for i in range(len(realRes)):
            res += (realRes[i] - expectedRes[i]) ** 2
        return res

    def saveToFile(self, filename: str):
        np.savez(filename, *self.b[1:], *self.w[1:])
    
    def loadFromFile(self, filename: str):
        npzfile = np.load(filename)
        print(len(npzfile.files))
        self.numOfLayers = len(npzfile.files) // 2 + 1

        self.z = [None]
        self.w = [None]
        self.b = [None]
                
        for i in range(1, self.numOfLayers):
            self.w.append(npzfile[f'arr_{self.numOfLayers - 2 + i}'])
            self.b.append(npzfile[f'arr_{i - 1}'])
        
        self.neuronCntPerLayers = [self.w[1].shape[1]]
        self.a = [np.ndarray(self.neuronCntPerLayers[0])]
        for i in range(1, self.numOfLayers):
            curCnt = self.b[i].shape[0]
            self.neuronCntPerLayers.append(curCnt)
            self.a.append(np.ndarray(curCnt))
            self.z.append(np.ndarray(curCnt))


            