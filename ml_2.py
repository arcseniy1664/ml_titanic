import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
df = pd.read_csv(r"C:\Users\Арсений\ml_titanic\processed_titanic1.csv")

dd=df.drop(columns=['Name','PassengerId','Cabin'])
#dd = df.select_dtypes(include='bool')
#dt = df.select_dtypes(include='float')
dt=dd
'''
dt= pd.DataFrame(df.data,columns=df.feature_names)



df['FoodCourt']=df.FoodCourt
print (df.head())
print("dframe:")
print (df.head())
print("variable X:")
print(X.head())
print("variable y:")
print(y.head())
'''

''''''
X=dd.drop(columns=['VIP_True'])
y=dd['VIP_True']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state =42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f'Log Accuracy:{accuracy:.2f}')
cm = confusion_matrix(y_test, y_pred)
#plt.show()
print(cm)

'''
print(X.head())

'''
a=dt.drop(columns=['Transported_True'])
b=dt['Transported_True']
a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.2, random_state =42)
model1 = LinearRegression()
model1.fit(a_train,b_train)
#b_pred = model1.predict(a_test)

b_pred = (model1.predict(a_test) > 0.5).astype(int)
b_test.astype(int)
accuracy1 = accuracy_score(b_test,b_pred)
print(f'Lin Accuracy:{accuracy1:.2f}')
mx = confusion_matrix(b_test, b_pred)
print(mx)