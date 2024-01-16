import json
from os import getenv,environ
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

environ['WANDB_API_KEY'] = getenv('WANDB_API_KEY')
environ['WANDB_NOTEBOOK_NAME'] = 'univariateModelling.ipynb'
# Login to wandb
wandb.login()

with open('sweepConfig.json','r',encoding='UTF-8') as jsonFile:
    sweepConfig = json.load(jsonFile)

sweep_id = wandb.sweep(sweepConfig, project='univarite stock opening price analysis')

df = pd.read_csv('aapl.csv')
scaler = MinMaxScaler()
scaledData = scaler.fit_transform(df['open'].values.reshape(-1, 1))

def createTrainTestSplit(data, sequenceLength):
    X, Y = [], []
    for i in range(sequenceLength, len(data)):
        X.append(data[i-sequenceLength:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)


feature, target = createTrainTestSplit(scaledData, 50)

xTrain,xTest,yTrain,yTest = train_test_split(feature,
                                                target,
                                                test_size=0.2,
                                                random_state=42,
                                                shuffle=False)

xTrainReshaped = np.reshape(xTrain, (xTrain.shape[0], 1, xTrain.shape[1]))
xTestReshaped = np.reshape(xTest, (xTest.shape[0], 1, xTest.shape[1]))

def train():
    
    wandb.init()

    
    model = Sequential()

    for i in range(wandb.config.lstm_layers):
        model.add(LSTM(wandb.config.neurons_per_layer,
                       return_sequences=True if i < wandb.config.lstm_layers - 1 else False))

    for i in range(wandb.config.dense_layers):
        model.add(Dense(wandb.config.neurons_per_layer))

    model.add(Dense(1))

    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    earlyStopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=2,
                                   mode='min',
                                   restore_best_weights=True)
    reduceLr = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.2,
                                 patience=5,
                                 verbose=2,
                                 mode='min')


    
    history = model.fit(xTrainReshaped, yTrain, validation_data=(xTestReshaped, yTest),
                        epochs=wandb.config.epochs, batch_size=wandb.config.batch_size,
                        callbacks=[WandbCallback(),earlyStopping,reduceLr])

    
    val_mse = history.history['val_mse'][-1]
    wandb.log({'val_mse': val_mse})

    
    predictions = model.predict(xTestReshaped)

    # Assuming 'dates', 'predictions', and 'yTest' are numpy arrays or lists
    dates = df['date'].values[-len(yTest):]  # Ensure 'dates' is correctly defined

    # Create a dataframe with the data
    data = {
        "Date": dates,
        "Predictions": predictions.flatten(),
        "Actual": yTest
    }

    dfPredicted = pd.DataFrame(data)
    dfPredicted["Date"] = pd.to_datetime(dfPredicted["Date"])

   
    wandb.log(
        {"predicted vs actual plot": wandb.plot.line_series(
            xs=list(dfPredicted.Date),
            ys=[list(dfPredicted.Predictions), list(dfPredicted.Actual)],
            keys=["Predicted", "Actual"],
            title="Actual and Predicted Values",
            xname="Date")})
    
    model.save(f"model_{wandb.run.name}.h5")
    wandb.finish()

wandb.agent(sweep_id, train)