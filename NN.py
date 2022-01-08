import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


# 匯出成csv
def output(filepath, data):
    df_SAMPLE = pd.DataFrame.from_dict(data)
    df_SAMPLE.to_csv(filepath, index=False)
    print('Success output to ' + filepath)


def RMSE(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


x_train = pd.read_csv('/Users/chuangray/Desktop/DL_Final/processed/x_train.csv', header=0, low_memory=False)
y_train = pd.read_csv('/Users/chuangray/Desktop/DL_Final/processed/y_train.csv', header=0, low_memory=False)
x_test = pd.read_csv('/Users/chuangray/Desktop/DL_Final/processed/x_test.csv', header=0, low_memory=False)
y_test = pd.read_csv('/Users/chuangray/Desktop/DL_Final/processed/y_test.csv', header=0, low_memory=False)
print('x_train.shape= {}, y_train.shape= {}'.format(x_train.shape, y_train.shape))
print('x_test.shape= {}, y_test.shape= {}'.format(x_test.shape, y_test.shape))

x_train.set_index('ID', inplace=True)
y_train.set_index('ID', inplace=True)
x_test.set_index('ID', inplace=True)
y_test.set_index('ID', inplace=True)

print('x_train.shape= {}, y_train.shape= {}'.format(x_train.shape, y_train.shape))
print('x_test.shape= {}, y_test.shape= {}'.format(x_test.shape, y_test.shape))

adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5)
sgd = SGD(learning_rate=1e-3)
model = Sequential()
model.add(Dense(1024, input_dim = x_train.shape[1], activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss=RMSE, optimizer=adam, metrics=[RMSE])
model.summary()
history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.005, verbose=1)

print(history.history.keys())
plt.figure(figsize=(15,8))
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

result = model.predict(x_test)
y_test['乳量'] = result
y_test.reset_index(inplace=True)
print(y_test.head(n=3))
output('/Users/chuangray/Desktop/DL_Final/result/submission(NN).csv', y_test)
