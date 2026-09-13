#script to load mnist dataset to csv files

import csv
import gzip
import struct
import urllib.request
import numpy as np

SOURCE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"

FILES = {
    "trainImages" : "train-images-idx3-ubyte.gz",
    "trainLabels" : "train-labels-idx1-ubyte.gz",
    "testImages"  : "t10k-images-idx3-ubyte.gz",
    "testLabels"  : "t10k-labels-idx1-ubyte.gz",
}

def download(filename):
    print("Downloading", filename, "...")
    urllib.request.urlretrieve(SOURCE_URL + filename, filename)

def loadImages(path):
    with gzip.open(path, "rb") as f:
        _, numImages, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(numImages, rows * cols)

def loadLabels(path):
    with gzip.open(path, "rb") as f:
        _, numLabels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def writeCsv(path, images, labels):
    normalized = images.astype(float) / 255.0   # scale 0-255 pixels to 0-1
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for pixels, label in zip(normalized, labels):
            writer.writerow([int(label)] + pixels.tolist())

for filename in FILES.values():
    download(filename)

print("loading images and labels")
xTrain = loadImages(FILES["trainImages"])
yTrain = loadLabels(FILES["trainLabels"])
xTest  = loadImages(FILES["testImages"])
yTest  = loadLabels(FILES["testLabels"])

print("writing to training.csv")
writeCsv("training.csv", xTrain, yTrain)

print("writing to testing.csv")
writeCsv("testing.csv", xTest, yTest)

print("successfully written data")