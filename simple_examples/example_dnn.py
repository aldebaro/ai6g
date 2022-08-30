'''
Example of a simple neural network.
Tested with Python 3.6 and tensorflow==1.14.0
'''
import numpy as np
from numpy.random import randint, standard_normal
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

#basic problem definitions
num_features = 4 #number of input features (dimension of input vector x)
num_classes = 30 #number of classes. Correct label is y \in {0, 1,..., num_classes}
num_train_examples = 100 # number of training examples
num_test_examples = 50 # number of test examples

#generate some random data and convert to one-hot encoding
X_train = standard_normal((num_train_examples, num_features))
y_train = randint(num_classes, size=num_train_examples)
y_train_onehot = to_categorical(y_train) #convert integers to one-hot encoding
X_test = standard_normal((num_test_examples, num_features))
y_test = randint(num_classes, size=num_test_examples)
y_test_onehot = to_categorical(y_test) #convert integers to one-hot encoding

#define the neural network topology
neural_net = Sequential() #use Keras functionality
neural_net.add(Dense(100, input_shape=(num_features,))) #dense layer
neural_net.add(Dropout(0.5)) #dropout to regularize the model
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dropout(0.5))
neural_net.add(Dense(num_classes, activation='softmax'))

#training stage
neural_net.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.01),
                    metrics=['accuracy','top_k_categorical_accuracy'])
history = neural_net.fit(X_train, y_train_onehot,
            batch_size=60, epochs=20)
#test stage
neural_net_outputs = neural_net.predict(X_test)
y_predicted = np.argmax(neural_net_outputs, axis=1) #indices

#display information
print(neural_net.summary()) #show topology
print('Accuracy = ', accuracy_score(y_test, y_predicted))

if False:
    print(X_train)
    print(y_train)
    print(y_train_onehot)
    print(y_predicted)