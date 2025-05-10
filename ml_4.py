import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
df = pd.read_csv(r"/Users/arenijserba/my_ml_project/my_ml_project/.venv/processed_titanic1.csv")

dd=df.drop(columns=['Name','PassengerId','Cabin'])
X=dd.drop(columns=['CryoSleep_True'])
y=dd['CryoSleep_True']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state =42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print("ACCURACY:")
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("Gradient Boosting:", accuracy_score(y_test, gb_pred))
print("F1 score:")
print("Random Forest:", f1_score(y_test, rf_pred, average='weighted'))
print("Gradient Boosting:", f1_score(y_test, gb_pred, average='weighted'))

rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='accuracy')

print("CROSS-VALIDATION ACCURACY:")
print("Random Forest:", rf_cv.mean())
print("Gradient Boosting:", gb_cv.mean())

models = ['Random Forest', 'Gradient Boosting']
accuracy = [accuracy_score(y_test, rf_pred), accuracy_score(y_test, gb_pred)]
plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()