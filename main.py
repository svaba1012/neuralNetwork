import numpy as np
from NeuralNetwork import NeuralNetwork

TRAINING_DATA_COUNT = 60000
TEST_DATA_COUNT = 10000

imagesFile = open('./mnist/train-images-idx3-ubyte', 'rb')
labelsFile = open('./mnist/train-labels-idx1-ubyte', 'rb')

labelsRaw = labelsFile.read()
labelsRaw = labelsRaw[8:]
imagesRaw = imagesFile.read()
imagesRaw = imagesRaw[16:]

labels = np.frombuffer(labelsRaw, dtype=np.uint8)
images = np.frombuffer(imagesRaw, dtype=np.uint8)
images = images.reshape(TRAINING_DATA_COUNT, 28*28)

imagesTestFile = open('./mnist/t10k-images-idx3-ubyte', 'rb')
labelsTestFile = open('./mnist/t10k-labels-idx1-ubyte', 'rb')

labelsTestRaw = labelsTestFile.read()
labelsTestRaw = labelsTestRaw[8:]
imagesTestRaw = imagesTestFile.read()
imagesTestRaw = imagesTestRaw[16:]

labelsTest = np.frombuffer(labelsTestRaw, dtype=np.uint8)
imagesTest = np.frombuffer(imagesTestRaw, dtype=np.uint8)
imagesTest = imagesTest.reshape(TEST_DATA_COUNT, 28*28)

images = images / 255
imagesTest = imagesTest / 255


neuralNet = NeuralNetwork([28*28, 16, 16, 10])
imagesTemp = images[:200]
labelsTemp = labels[:200]
neuralNet.setTestData(imagesTest, labelsTest)

avgCost = neuralNet.test()
print("Initital avg cost:", avgCost)

neuralNet.saveToFile(f'neuralNet0.npz')

for epoch in range(2):
    for i in range(120):
        imagesTemp = images[i*500:(i+1)*500]
        labelsTemp = labels[i*500:(i+1)*500]
        neuralNet.setTrainingData(imagesTemp, labelsTemp)
        neuralNet.train(learningRate=0.5, batchSize=100)
        avgCost = neuralNet.test()
        print("After,", (i + 1) * 500, "samples training,", epoch, "epoch:", avgCost)
    neuralNet.saveToFile(f'neuralNet{epoch + 1}.npz')

