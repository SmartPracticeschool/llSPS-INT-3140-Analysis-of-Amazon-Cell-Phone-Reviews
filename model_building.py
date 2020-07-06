
"Import The Dataset"
import numpy as np
import pandas as pd
dataset = pd.read_csv("amazon_review.csv", sep = ',', quoting = 1)
dataset = dataset.iloc[:, 2:8]
dataset = dataset.drop("date", axis = 1)
dataset['title'].fillna(value = "No Title", inplace = True)

"Remove Punctuations, Numbers"
import re

#For Stemming
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
newdata = []

for i in range(0, 67986): 
    title_review = re.sub('[^a-zA-z]', " ", dataset['title'][i])
    
    "Convert Each Number Into Its Lower Case"
    title_review = title_review.lower()
    title_review = title_review.split() #converting into list

    "Apply Stemming"
    title_review = [ps.stem(word) for word in title_review if not word in set(stopwords.words('english'))]
    title_review = ' '.join(title_review)
    newdata.append(title_review)
    


"Splitting Data Into Traning And Test Set"
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
cv.fit_transform(newdata)
x = cv.fit_transform(newdata).toarray()
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

"Importing the libraries"
import pickle
import keras
from keras.models import Sequential 
from keras.layers import Dense 
"Initializing the model"
model = Sequential()

"Adding input layer"
model.add(Dense( units= 1500,init = 'uniform',activation = 'relu'))

"Adding the hidden layer"
model.add(Dense(units =750,init ='uniform',activation = 'relu'))

"Adding the output Layer"
model.add(Dense(units=1 ,init = 'uniform',activation = 'sigmoid'))

"Configuring the learning process"
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

"Training the model"
model.fit(x_train,y_train,batch_size =32,epochs=100)

pickle.dump(cv, open("CountVectorizer.pickle", "wb"))

"Save The Model"
model.save('reviews.h5')





