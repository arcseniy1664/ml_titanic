from operator import countOf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
'''
'''
# Генерация тестовых данных
np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = []
for row in X:
    c = np.count_nonzero(row == 1)
    if c > 6:
        Y.append([1, 0])
    else:
        Y.append([0, 1])

Y = np.array(Y)
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
   keras.layers.Dense(12, input_shape=(12,)),
   keras.layers.Dense(8, activation='sigmoid'),
   keras.layers.Dense(2, activation='softmax')
])

# Компиляция модели
early_stop = EarlyStopping(monitor='val_loss', patience=10) #остановка если точность не увеличивается
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )

# Обучение модели
history = model.fit(X_train, y_train, epochs=60, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])



# Оцениваем качество на тестовой выборке
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))

# График изменения функции ошибки
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


























'''
# Логистическая регрессия

lr = LogisticRegression()
lr.fit(x_tr, y_tr)
y_pred_lr = lr.predict(x_t)
print("Logistic Regression Accuracy:", accuracy_score(y_t, y_pred_lr))


# Случайный лес
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_tr, y_tr)
y_pred_rf = rf.predict(x_t)
print("Random Forest Accuracy:", accuracy_score(y_t, y_pred_rf))
'''















#
'''
y_true_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print("Precision:",precision_score(y_true_class, y_pred, zero_division=0))
print("Recall:",recall_score(y_true_class,y_pred,zero_division=0))
print("F1:",f1_score(y_true_class,y_pred,zero_division=0))



print(f" Precision: {precision:.2f}")
print(f" Recall:    {recall:.2f}")
print(f" F1-score:  {f1:.2f}")
'''