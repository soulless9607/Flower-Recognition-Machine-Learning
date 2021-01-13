from utils import load_data
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plot 

from sklearn.model_selection import train_test_split

(feature, labels) = load_data() 

x_train, x_test, y_train, y_test = train_test_split(feature,labels,test_size=0.1) 
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = tf.keras.models.load_model('mymodel.h5')
#mode.evaluate(x_test, y_test, verbose = 1)

prediction = model.predict(x_test)
plot.figure(figsize=(9,9))
for i in range(9): #nueve imágenes 
	plot.subplot(3,3,i+1)
	plot.imshow(x_test[i])
	plot.xlabel('Actual:'+categories[y_test[i]]+'\n'+'Predicted:'+ categories[np.argmax(prediction[i])])
	plot.xticks([])
plot.show() 

