import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] =20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
df.head()

#what we are going to do is to arrange the stocks data or data point values by the order of dates 
df["Date"] = pd.to_datetime(df.Date,format = "%Y-%m-%d")
df.index = df["Date"]
plt.figure(figsize = (13,8))
plt.plot(df["Close"],label = "closing price of the stock")
#now we will take a new dataset which will store only the information containing closing stock with its date
data= df.sort_index(ascending=True,axis = 0)
new_dataset = pd.DataFrame(index=range(0,len(df)),columns = ["Date","Close"])
for i in range(0,len(df)):
  new_dataset["Date"][i] = data["Date"][i]
  new_dataset["Close"][i] = data["Close"][i]
#new_dataset["Date"] = pd.to_datetime(new_dataset.Date,format = "%Y-%m-%d")
new_dataset.index = new_dataset["Date"]
plt.figure(figsize = (13,8))
plt.plot(new_dataset["Close"],label = "closing price of the stock")

scaler = MinMaxScaler(feature_range =(0,1))
final_dataset = new_dataset.values
train_data = final_dataset[0:987,:]
valid_data = final_dataset[987:,:]

#final dataset is an array of the values of new dataset 
#so we are converting our dataframe into array of values 
new_dataset
new_dataset.drop("Date",axis =1,inplace = True)
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(new_dataset)

x_train_data ,y_train_data= [],[]
for i in range(60,len(train_data)):
  x_train_data.append(scaled_data[i-60:i,0])
  y_train_data .append(scaled_data[i,0])
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
x_train_data

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

train_data=new_dataset[:987]
valid_data=new_dataset[987:]
valid_data['Predictions']=predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])