# Neural Network

Neural network build from scratch in python using only numpy. It provides cli to interact with neural network.
Thanks to [3blue1brown](https://www.3blue1brown.com/) astonishing youtube playlist about [neural network](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) I tried to make simple neural network from scratch, implementing my own backpropagation.
To visualize results I used [CS50 code](https://github.com/KevinLiTian/Harvard_CS50_AI/blob/main/5.Neural_Networks/examples/digits/recognition.py) to enable user to draw digits.

### Usage

> Run python3 main.py -h
```
usage: NeuralNetwork [-h] [-f FILENAME] [-e EPOCH] [-bs BATCHSIZE] [-lr LEARNINGRATE] command

Train, test or use neural network

positional arguments:
command

options:
-h, --help show this help message and exit
-f FILENAME, --filename FILENAME
Name of the file that will be used to save or load neural network, default
'neuralNet.npz'
-e EPOCH, --epoch EPOCH
Number of epoch for neural network training, deafault 1
-bs BATCHSIZE, --batchSize BATCHSIZE
Size of batch for neural network training, default 10
-lr LEARNINGRATE, --learningRate LEARNINGRATE
Learning rate for neural network training, default 1.0

Neural network built from scratch in python using only numpy
```
### Ways to improve
- Currently backpropagation is calculated iteratively looping through each element. Implementing backpropagation using matrixes will speed up training process
- ...
