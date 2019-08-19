#importing important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset=pd.read_csv('pubg.csv')
#visualising dataset
dataset.head()
dataset.info()
sns.distplot(dataset['walkDistance'])
sns.distplot(dataset['kills'])
sns.distplot(dataset['rideDistance'])
sns.distplot(dataset['matchDuration'])
sns.distplot(dataset['killStreaks'])
sns.distplot(dataset['revives'])
dataset.hist('weaponsAcquired',range=(0,20))
sns.scatterplot(x=dataset['winPlacePerc'],y=dataset["kills"])
#removal of unwanted data 
dataset.drop(dataset[dataset['kills']>39].index,inplace=True)
dataset.drop(dataset[dataset['rideDistance']>15000].index,inplace=True)
dataset['totaldist']=dataset['rideDistance']+dataset['walkDistance']+dataset['swimDistance']
dataset['killwithoutmoving']=((dataset['kills']>0)&(dataset['totaldist']==0))
dataset.drop(dataset[dataset['killwithoutmoving']==True].index,inplace=True)
#spliting of independent and dependent parameters
X=dataset.iloc[:,3:28].values
y=dataset.iloc[:,28].values
#Encoding using onehotencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 12] = labelencoder_X_1.fit_transform(X[:, 12])
onehotencoder = OneHotEncoder(categorical_features = [12])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #for overcome the onehotencoder trap
#spliting of traing and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)
#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu', input_dim = 39))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(output_dim =80, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
#traing of ANN
history=classifier.fit(X_train, y_train, batch_size = 254, nb_epoch = 10,validation_split=0.20)
#prediction
prediction=classifier.predict(X_test)
#comparision of predicted and real 
plt.plot(y_test[100:150],color='red',label="real")
plt.plot(prediction[100:150],color='blue',label="predicted")
plt.legend()
plt.show()
#epochs vs loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel("Epochs")
plt.legend(['train','validation'],loc='upper left')
plt.show()
