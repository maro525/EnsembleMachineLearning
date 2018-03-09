from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(100)
n_of_neurons=100
model = Sequential()
model.add(Dense(input_dim=2,output_dim=n_of_neurons,activation="relu"))
model.add(Dense(n_of_neurons,activation="relu"))
model.add(Dense(n_of_neurons,activation="relu"))
model.add(Dense(n_of_neurons,activation="relu"))
model.add(Dense(n_of_neurons,activation="relu"))
model.add(Dense(n_of_neurons,activation="relu"))
model.add(Dense(1))

adam=Adam(lr=1e-5)

data=pd.read_csv('../data/ice.csv')
x=data[['temp','street']]
xnp=x.as_matrix()
y=data['ice']
ynp=y.as_matrix()

model.compile(loss='mse',optimizer='adam')
hist=model.fit(xnp,ynp,nb_epoch=20000,verbose=1,shuffle=True)
model.evaluate(xnp,ynp,verbose=0)
print("score: ",r2_score(ynp,model.predict_proba(xnp)))
#print("y: ",ynp)
#print(model.predict_proba(xnp))
