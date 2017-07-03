# %% import
import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    os.chdir("./Reezocar")
except FileNotFoundError:
    print("Current working directory : ", os.getcwd())

print("File size :")
for f in os.listdir('./input'):
     print(f + '   ' + str(round(os.path.getsize('./input/' + f)/1000000, 2)) + 'MB')

# %% Load data

df_import = pd.read_csv('./input/toyota.csv')

# %% function definition

def data_cleaning(df):
    df_train = df.drop(df[df.Price > 30000].index)

    # Feature Engineering

    fuel_type = pd.get_dummies(df_train.FuelType)
    # print(fuel_type.head())
    # df_train_x = df_train_x.drop(["FuelType"], axis=1)
    df_train = df_train.drop(["FuelType", "CC", "MetColor", "Doors"], axis=1)

    df_train = pd.concat([df_train, fuel_type], axis=1)
    return df_train

def baseline_model():
    model = Sequential()

    # MODEL TUTORIAL KERAS
    # model.add(Dense(units=64, input_dim=8))
    # model.add(Activation('relu'))
    # model.add(Dense(units=10))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


    # MODEL FROM GITHUB WITH LSTM
    # model.add(LSTM(24, input_shape=(1003, 8), return_sequences=True, implementation=2))
    # model.add(TimeDistributed(Dense(1)))
    # model.add(AveragePooling1D())
    # model.add(Flatten())
    # model.add(Dense(2, activation='softmax'))

    # MODEL FROM TUTORIAL mlmastery.com
    model.add(Dense(64, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')

    return model

def model_training(df):
    df_y = df.Price
    df_x = df.drop(["Price"], axis=1)
    np_x = df_x.as_matrix()
    np_y = df_y.as_matrix()

    train_x, valid_x, train_y, valid_y = train_test_split(df_x, df_y, train_size=.7)
    np_train_x = train_x.as_matrix()
    np_train_y = train_y.as_matrix()
    np_valid_x = valid_x.as_matrix()
    np_valid_y = valid_y.as_matrix()

    # Fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=5, batch_size=32, verbose=0)))
    pipeline = Pipeline(estimators)
    # Evaluate model with 10-fold cross validation
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, np_x, np_y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


    # model = baseline_model()
    # model.fit(np_train_x, np_train_y, epochs=10, batch_size=5)
    # loss_and_metric = model.evaluate(np_valid_x, np_valid_y, batch_size=128)

def main():
    df_train = data_cleaning(df_import)
    df_train.Price.mean()

    # print(df_train)
    model = model_training(df_train)




# %% Launch main

if __name__ ==  '__main__':
    main()
