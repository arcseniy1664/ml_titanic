import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)
df = pd.read_csv(r"/Users/arenijserba/my_ml_project/my_ml_project/.venv/processed_titanic1.csv")
dd=df.drop(columns=['Name','PassengerId','Cabin'])

X=dd.drop(columns=['CryoSleep_True'])
y=dd['CryoSleep_True']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state =42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(10, 8))
plot_tree(clf, filled=True)
plt.show()
