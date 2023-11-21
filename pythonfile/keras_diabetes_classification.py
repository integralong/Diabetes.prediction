from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy

# random seed for reproducibility
numpy.random.seed(2)

# loading load prima indians diabetes dataset, past 5 years of medical history 
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:8]
Y = dataset[:,8]

# split X, Y into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu')) # input layer requires input_dim param
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

# compile the model, adam gradient descent (optimized)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# call the function to fit to the data (training the network)
history = model.fit(x_train, y_train, epochs = 1000, batch_size=20, validation_data=(x_test, y_test), verbose = 0)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


import matplotlib.pyplot as plt

#get the figure and the axes
fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, sharey = False, figsize = (10, 5))

#draw accuracy of model
ax0.plot(history.history['accuracy'])
ax0.set(title='model accuracy', xlabel = 'epoch', ylabel ='accuracy')

#draw loss of model
ax1.plot(history.history['loss'])
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')

plt.show()


#------patient_1-----
# 가상의 환자 데이터 입력
patient_1 = numpy.array([[0, 137, 90, 35, 168, 43.1, 2.288, 33]])

# 모델로 예측하기
prediction = model.predict(patient_1)

# 예측결과 출력하기
print(prediction * 100)

