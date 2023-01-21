from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import  ImageDataGenerator
# Creating the CNN pipeline
convolution = Sequential()

# creating the convolution with 32 Kernel and input shape of 64*64*3-color and then Relu activation
convolution.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# Add the Max pooling with 2*2 Matrix
convolution.add(MaxPool2D(2, 2))
# After convolution flatted the matrix into array of input
convolution.add(Flatten())

# Input is created and then create the neural network
# create the hidden layer with 2neurons

convolution.add(Dense(units=2, activation='relu'))

# compile the neural network
convolution.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# preprocessing the image
train = ImageDataGenerator(rescale=1./255)
test = ImageDataGenerator(rescale=1./255)

training_set = train.flow_from_directory("/C/Users/navan/datascience/ML models/Deep_Learning/food_classfication/train", batch_size  =32, target_size=(64, 64), class_mode='binary')
testing_set = test.flow_from_directory("/C/Users/navan/datascience/ML models/Deep_Learning/food_classfication/test", batch_size=32, target_size=(64, 64), class_mode='binary')

# create and fir the model

model = convolution.fit_generator(training_set, steps_per_epoch=8000, epochs=1, validation_data=testing_set, validation_steps =2000)
convolution.save('model.hs')
print("saved to disk")



