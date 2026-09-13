import csv
import numpy as np
from NeuralNetwork import *

def toVector(label, numClasses=10):
    vector = np.zeros(numClasses)
    vector[int(label)] = 1
    return vector

def createBatches(trainingData, batchSize):
    np.random.shuffle(trainingData)
    return [trainingData[i:i+batchSize] for i in range(0, len(trainingData), batchSize)]

trainingLines = []
with open("training.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        trainingLines.append(row)
trainingData = np.array(trainingLines, dtype=float)

testingLines = []
with open("testing.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        testingLines.append(row)
testingData = np.array(testingLines, dtype=float)

neuralNetwork = NeuralNetwork([784, 128, 64, 10])

learningRate = 0.1

print("training network with learning rate",learningRate,"...")
for iteration in range(500):
    #----training----
    batches = createBatches(trainingData, 20)

    for batch in batches:
        featuresList = []
        targetsList  = []
        for row in batch:
            label, features = row[0], row[1:]
            featuresList.append(features)
            targetsList.append(toVector(label))
        neuralNetwork.trainBatch(featuresList, targetsList, learningRate)

    learningRate *= 0.99

    if (iteration+1) % 50 == 0:
        #----testing----
        correct   = 0
        incorrect = 0
        for row in testingData:
            label, features = row[0], row[1:]
            prediction = neuralNetwork.predict(features)
            if prediction == label:
                correct += 1
            else:
                incorrect += 1

        acc = correct / (correct + incorrect)
        print("iteration",iteration+1,"accuracy:",acc)

print("\nFinal model accuracy:",acc)