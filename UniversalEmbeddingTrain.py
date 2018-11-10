from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import numpy
import csv

deenEmbedFile = "de-en-EmbedSmall.csv"
X = []
Y = []
#get data
with open(deenEmbedFile, 'r', encoding='utf-8') as csvf:
    csvreader = csv.reader(csvf,delimiter=',')
    i=0
    for row in csvreader:
        X.append(list(map(float,row[1].split())))
        Y.append(list(map(float,row[3].split())))
csvf.close()        
#make X and Y as arrays

X = np.asarray(X)
Y = np.asarray(Y)

#build the model
model = Sequential()
model.add(Dropout(0.3, input_shape=(300,)))
model.add(Dense(300, input_dim=300, activation='relu'))
model.add(Dense(300, activation='tanh'))

#compile the model
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.8, nesterov=True), metrics=['accuracy'])

#fit the model
model.fit(X,Y, epochs=10, batch_size=32)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
