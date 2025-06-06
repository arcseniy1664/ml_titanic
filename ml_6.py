import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt  # Добавлено для визуализации

# 1. Загрузка данных
train = pd.read_csv("sign_mnist_train.csv")  # Обучающая выборка
test = pd.read_csv("sign_mnist_test.csv")  # Тестовая выборка

# 2. Подготовка данных
num_classes = 25  # 25 классов жестов (A-Y)

# Преобразование обучающих данных
X_train = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(train['label'], num_classes=num_classes)

# Преобразование тестовых данных
X_test = test.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_test = tf.keras.utils.to_categorical(test['label'], num_classes=num_classes)

# 3. Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 4. Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Обучение модели
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=2,
          batch_size=32)



def predict_gesture(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    sample = X_test[5].reshape(1, 28, 28, 1)
    prediction = model.predict(sample)
    print(f'Предсказанный класс: {np.argmax(prediction)}')  # индекс с самой высокой вероятностью
    print(f'Настоящий класс: {np.argmax(y_test[5])}')

    plt.imshow(X_test[5].reshape(28, 28), cmap='gray')
    plt.title(f'Метка класса: {np.argmax(y_test[5])}')
    plt.axis('off')
    plt.show()

    return class_idx
predict_gesture("amer_sign2.png")

