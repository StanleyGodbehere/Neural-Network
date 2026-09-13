import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dSigmoid(x):
    return x * (1 - x)

def leakyReLU(x):
    return np.where(x > 0, x, x * 0.01)

def dLeakyReLU(x):
    return np.where(x > 0, 1, 0.01)

class Layer:
    def __init__(self, numInputs, size, activation):
        self.numInputs  = numInputs
        self.size       = size
        self.activation = activation

        limit = np.sqrt(2 / numInputs) if activation == "reLU" else np.sqrt(6 / (numInputs + size))
        self.weights            = np.random.uniform(-limit, limit, (size, numInputs))
        self.biases             = np.zeros(size)
        self.inputs             = []
        self.linearCombinations = []
        self.outputs            = []

    def processInputs(self, inputs):
        self.inputs                 = np.array(inputs)
        self.linearCombinations     = np.dot(self.weights, self.inputs) + self.biases
        
        if self.activation == "reLU":
            self.outputs = leakyReLU(self.linearCombinations)
        else:
            self.outputs = sigmoid(self.linearCombinations)
        return self.outputs
    
class NeuralNetwork:
    def __init__(self, layerSizes):
        self.layers = []
        print("Created neural network")
        for i in range(len(layerSizes) - 1):
            activation = "sigmoid" if i == len(layerSizes) - 2 else "reLU"
            self.layers.append(Layer(layerSizes[i], layerSizes[i+1], activation))
            print("Layer",i,":",layerSizes[i],"inputs,",layerSizes[i+1],"neurons (",activation,")")
        print()

    def processValues(self, values):
        for layer in self.layers:
            values = layer.processInputs(values)
        return values
    
    def predict(self, inputs):
        output = self.processValues(inputs)[0]
        return 1 if output > 0.5 else 0
    
    def cost(self, inputs, expectedOutputs):
        outputs   = self.processValues(inputs)
        totalCost = 0
        
        for i in range(len(outputs)):
            y = expectedOutputs[i]
            p = outputs[i]
            totalCost += (-(y*np.log(p)+(1-y)*np.log(1-p)))
        return totalCost / len(outputs)
    
    def trainBatch(self, inputsList, expectedOutputsList, learningRate):
        accumulatedDeltas = None
        accumulatedInputs = None

        for (inputs, expectedOutputs) in zip(inputsList, expectedOutputsList):
            deltas, layerInputs = self.getDeltas(inputs, expectedOutputs)
            
            if accumulatedDeltas is None:
                accumulatedDeltas = deltas
                accumulatedInputs = layerInputs
            else:
                for i in range(len(accumulatedDeltas)):
                    accumulatedDeltas[i] += deltas[i]
                    accumulatedInputs[i] += layerInputs[i]
        
        batchSize = len(inputsList)
        for i in range(len(self.layers)):
            layer        = self.layers[i]
            delta        = accumulatedDeltas[i] / batchSize
            averageInput = accumulatedInputs[i] / batchSize
            
            layer.weights -= learningRate * np.outer(delta, averageInput)
            layer.biases  -= learningRate * delta

    def getDeltas(self, inputs, expectedOutputs):
        outputs         = self.processValues(inputs)
        expectedOutputs = np.array(expectedOutputs)
        deltas          = [outputs - expectedOutputs]
        layerInputs     = [layer.inputs for layer in self.layers]
        
        for i in reversed(range(len(self.layers) - 1)):
            current  = self.layers[i + 1]
            previous = self.layers[i]

            delta = np.dot(current.weights.T, deltas[-1]) * dLeakyReLU(previous.linearCombinations)
            deltas.append(delta)

        deltas.reverse() 
        return deltas, layerInputs
    
    def trainSingle(self, inputs, expectedOutputs, learningRate):
        outputs         = self.processValues(inputs)
        expectedOutputs = np.array(expectedOutputs)
        deltas          = [outputs - expectedOutputs]
        
        for i in reversed(range(len(self.layers) - 1)):
            current  = self.layers[i + 1]
            previous = self.layers[i]

            delta = np.dot(current.weights.T, deltas[-1]) * dLeakyReLU(previous.linearCombinations)
            deltas.append(delta)

        deltas.reverse() 
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            delta = deltas[i]
            
            layer.weights -= learningRate * np.outer(delta, layer.inputs)
            layer.biases  -= learningRate * delta