import numpy as np
import argparse
from NeuralNetwork import NeuralNetwork
from gui import startGui

TRAINING_DATA_COUNT = 60000
TEST_DATA_COUNT = 10000

def trainNeuralNet(filename, epoch, batchSize, learningRate):
    '''
    Train neural network and save it to file
    '''
    imagesFile = open('./mnist/train-images-idx3-ubyte', 'rb')
    labelsFile = open('./mnist/train-labels-idx1-ubyte', 'rb')

    labelsRaw = labelsFile.read()
    labelsRaw = labelsRaw[8:]
    imagesRaw = imagesFile.read()
    imagesRaw = imagesRaw[16:]

    labels = np.frombuffer(labelsRaw, dtype=np.uint8)
    images = np.frombuffer(imagesRaw, dtype=np.uint8)
    images = images.reshape(TRAINING_DATA_COUNT, 28*28)

    images = images / 255

    images = images[:100]
    labels = labels[:100]

    neuralNet = NeuralNetwork([28*28, 16, 16, 10])
    neuralNet.setTrainingData(images, labels)
    print("Initializing neural network");

    for epochId in range(epoch):
        print("Starting epoch", epochId + 1, "of", epoch, "epoch")
        neuralNet.train(batchSize, learningRate)
    neuralNet.saveToFile(filename)

    print("Training finished")

def testNeuralNet(filename):
    '''
    Load neural network from file and test it agains test data
    '''
    print("Loading neural network from:", filename)
    imagesTestFile = open('./mnist/t10k-images-idx3-ubyte', 'rb')
    labelsTestFile = open('./mnist/t10k-labels-idx1-ubyte', 'rb')

    labelsTestRaw = labelsTestFile.read()
    labelsTestRaw = labelsTestRaw[8:]
    imagesTestRaw = imagesTestFile.read()
    imagesTestRaw = imagesTestRaw[16:]

    labelsTest = np.frombuffer(labelsTestRaw, dtype=np.uint8)
    imagesTest = np.frombuffer(imagesTestRaw, dtype=np.uint8)
    imagesTest = imagesTest.reshape(TEST_DATA_COUNT, 28*28)

    imagesTest = imagesTest / 255

    neuralNet = NeuralNetwork([])
    neuralNet.loadFromFile(filename)
    neuralNet.setTestData(imagesTest, labelsTest)
    print("Testing...")
    avgCost = neuralNet.test()
    print("Average cost:", avgCost)

# Initialize parser
parser = argparse.ArgumentParser(
                    prog='NeuralNetwork',
                    description='Train, test or use neural network',
                    epilog='Neural network built from scratch in python using only numpy')

parser.add_argument('command')

parser.add_argument("-f", "--filename", help = "Name of the file that will be used to save or load neural network, default 'neuralNet.npz'", default="neuralNet.npz")

parser.add_argument("-e", "--epoch", help = "Number of epoch for neural network training, default 1", default=1)
parser.add_argument("-bs", "--batchSize", help = "Size of batch for neural network training, default 10", default=10)
parser.add_argument("-lr", "--learningRate", help = "Learning rate for neural network training, default 1.0", default=1.0)

# Read arguments from command line
args = parser.parse_args()

if args.command == "train":
    print("Training neural net")
    trainNeuralNet(args.filename, args.epoch, args.batchSize, args.learningRate)
elif args.command == "test":
    testNeuralNet(args.filename)
elif args.command == "gui":
    startGui(args.filename)
else:
    print("Unsupported command")
    exit(1)










