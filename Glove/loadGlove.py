import numpy as np
from gensim.models import KeyedVectors

def loadGloveModel(gloveFile):
    print("loading glove model")
    f = open(gloveFile, 'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word]= embedding
    print("done. ", len(model), " words loaded")
    return model


def loadGloveUsingGenSim(gloveFile):
    print("loading model..")
    model_2 = Word2Vec

model = loadGloveModel("e:/Glove/glove.6B.50d.txt")
print("the is ", model["the"])

#print("Checking to see what comes for ORD and chicago - ORD",model["ORD"], " Chicago - ", model["Chicago"])