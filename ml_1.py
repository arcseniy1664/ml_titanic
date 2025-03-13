import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
dd = pd.read_csv(r"C:\Users\Арсений\myenv\train.csv")
missing = dd.isnull().sum()


#заполнение train.csv
dd["HomePlanet"] = dd["HomePlanet"].fillna(dd["HomePlanet"].mode()[0])
dd["CryoSleep"] = dd["CryoSleep"].fillna(dd["CryoSleep"].median())
dd["Cabin"] = dd["Cabin"].fillna(dd["Cabin"].mode()[0])
dd["Destination"] = dd["Destination"].fillna(dd["Destination"].mode()[0])
dd["Age"] = dd["Age"].fillna(dd["Age"].mode()[0])
dd["VIP"] = dd["VIP"].fillna(dd["VIP"].median())
dd["RoomService"] = dd["RoomService"].fillna(dd["RoomService"].mode()[0])
dd["FoodCourt"] = dd["FoodCourt"].fillna(dd["FoodCourt"].mode()[0])
dd["ShoppingMall"] = dd["ShoppingMall"].fillna(dd["ShoppingMall"].mode()[0])
dd["Spa"] = dd["Spa"].fillna(dd["Spa"].mode()[0])
dd["VRDeck"] = dd["VRDeck"].fillna(dd["VRDeck"].mode()[0])
missing = dd.isnull().sum()




scaler = MinMaxScaler()
dd["Age"] = scaler.fit_transform(dd[["Age"]])
dd["RoomService"] = scaler.fit_transform(dd[["RoomService"]])
dd["FoodCourt"] = scaler.fit_transform(dd[["FoodCourt"]])
dd["ShoppingMall"] = scaler.fit_transform(dd[["ShoppingMall"]])
dd["Spa"] = scaler.fit_transform(dd[["Spa"]])
dd["VRDeck"] = scaler.fit_transform(dd[["VRDeck"]])



dd = pd.get_dummies(dd, columns=['HomePlanet'], drop_first=True)
dd = pd.get_dummies(dd, columns=['CryoSleep'], drop_first=True)
dd = pd.get_dummies(dd, columns=['Destination'], drop_first=True)
dd = pd.get_dummies(dd, columns=['VIP'], drop_first=True)
dd = pd.get_dummies(dd, columns=['RoomService'], drop_first=True)
dd = pd.get_dummies(dd, columns=['FoodCourt'], drop_first=True)
dd = pd.get_dummies(dd, columns=['ShoppingMall'], drop_first=True)
dd = pd.get_dummies(dd, columns=['Spa'], drop_first=True)
dd = pd.get_dummies(dd, columns=['VRDeck'], drop_first=True)
dd = pd.get_dummies(dd, columns=['Transported'], drop_first=True)

dd.to_csv("processed_titanic.csv", index=False)







