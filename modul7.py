# ============================================================
# CELL 1 - Import library
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CELL 2 - Load dataset (tanpa CSV/URL, data hardcode)
# ============================================================
data = np.array([
    [5.1,3.5,1.4,0.2,'Iris-setosa'], [4.9,3.0,1.4,0.2,'Iris-setosa'],
    [4.7,3.2,1.3,0.2,'Iris-setosa'], [4.6,3.1,1.5,0.2,'Iris-setosa'],
    [5.0,3.6,1.4,0.2,'Iris-setosa'], [5.4,3.9,1.7,0.4,'Iris-setosa'],
    [4.6,3.4,1.4,0.3,'Iris-setosa'], [5.0,3.4,1.5,0.2,'Iris-setosa'],
    [4.4,2.9,1.4,0.2,'Iris-setosa'], [4.9,3.1,1.5,0.1,'Iris-setosa'],
    [5.4,3.7,1.5,0.2,'Iris-setosa'], [4.8,3.4,1.6,0.2,'Iris-setosa'],
    [4.8,3.0,1.4,0.1,'Iris-setosa'], [4.3,3.0,1.1,0.1,'Iris-setosa'],
    [5.8,4.0,1.2,0.2,'Iris-setosa'], [5.7,4.4,1.5,0.4,'Iris-setosa'],
    [5.4,3.9,1.3,0.4,'Iris-setosa'], [5.1,3.5,1.4,0.3,'Iris-setosa'],
    [5.7,3.8,1.7,0.3,'Iris-setosa'], [5.1,3.8,1.5,0.3,'Iris-setosa'],
    [5.4,3.4,1.7,0.2,'Iris-setosa'], [5.1,3.7,1.5,0.4,'Iris-setosa'],
    [4.6,3.6,1.0,0.2,'Iris-setosa'], [5.1,3.3,1.7,0.5,'Iris-setosa'],
    [4.8,3.4,1.9,0.2,'Iris-setosa'], [5.0,3.0,1.6,0.2,'Iris-setosa'],
    [5.0,3.4,1.6,0.4,'Iris-setosa'], [5.2,3.5,1.5,0.2,'Iris-setosa'],
    [5.2,3.4,1.4,0.2,'Iris-setosa'], [4.7,3.2,1.6,0.2,'Iris-setosa'],
    [4.8,3.1,1.6,0.2,'Iris-setosa'], [5.4,3.4,1.5,0.4,'Iris-setosa'],
    [5.2,4.1,1.5,0.1,'Iris-setosa'], [5.5,4.2,1.4,0.2,'Iris-setosa'],
    [4.9,3.1,1.5,0.2,'Iris-setosa'], [5.0,3.2,1.2,0.2,'Iris-setosa'],
    [5.5,3.5,1.3,0.2,'Iris-setosa'], [4.9,3.6,1.4,0.1,'Iris-setosa'],
    [4.4,3.0,1.3,0.2,'Iris-setosa'], [5.1,3.4,1.5,0.2,'Iris-setosa'],
    [5.0,3.5,1.3,0.3,'Iris-setosa'], [4.5,2.3,1.3,0.3,'Iris-setosa'],
    [4.4,3.2,1.3,0.2,'Iris-setosa'], [5.0,3.5,1.6,0.6,'Iris-setosa'],
    [5.1,3.8,1.9,0.4,'Iris-setosa'], [4.8,3.0,1.4,0.3,'Iris-setosa'],
    [5.1,3.8,1.6,0.2,'Iris-setosa'], [4.6,3.2,1.4,0.2,'Iris-setosa'],
    [5.3,3.7,1.5,0.2,'Iris-setosa'], [5.0,3.3,1.4,0.2,'Iris-setosa'],

    [7.0,3.2,4.7,1.4,'Iris-versicolor'], [6.4,3.2,4.5,1.5,'Iris-versicolor'],
    [6.9,3.1,4.9,1.5,'Iris-versicolor'], [5.5,2.3,4.0,1.3,'Iris-versicolor'],
    [6.5,2.8,4.6,1.5,'Iris-versicolor'], [5.7,2.8,4.5,1.3,'Iris-versicolor'],
    [6.3,3.3,4.7,1.6,'Iris-versicolor'], [4.9,2.4,3.3,1.0,'Iris-versicolor'],
    [6.6,2.9,4.6,1.3,'Iris-versicolor'], [5.2,2.7,3.9,1.4,'Iris-versicolor'],
    [5.0,2.0,3.5,1.0,'Iris-versicolor'], [5.9,3.0,4.2,1.5,'Iris-versicolor'],
    [6.0,2.2,4.0,1.0,'Iris-versicolor'], [6.1,2.9,4.7,1.4,'Iris-versicolor'],
    [5.6,2.9,3.6,1.3,'Iris-versicolor'], [6.7,3.1,4.4,1.4,'Iris-versicolor'],
    [5.6,3.0,4.5,1.5,'Iris-versicolor'], [5.8,2.7,4.1,1.0,'Iris-versicolor'],
    [6.2,2.2,4.5,1.5,'Iris-versicolor'], [5.6,2.5,3.9,1.1,'Iris-versicolor'],
    [5.9,3.2,4.8,1.8,'Iris-versicolor'], [6.1,2.8,4.0,1.3,'Iris-versicolor'],
    [6.3,2.5,4.9,1.5,'Iris-versicolor'], [6.1,2.8,4.7,1.2,'Iris-versicolor'],
    [6.4,2.9,4.3,1.3,'Iris-versicolor'], [6.6,3.0,4.4,1.4,'Iris-versicolor'],
    [6.8,2.8,4.8,1.4,'Iris-versicolor'], [6.7,3.0,5.0,1.7,'Iris-versicolor'],
    [6.0,2.9,4.5,1.5,'Iris-versicolor'], [5.7,2.6,3.5,1.0,'Iris-versicolor'],
    [5.5,2.4,3.8,1.1,'Iris-versicolor'], [5.5,2.4,3.7,1.0,'Iris-versicolor'],
    [5.8,2.7,3.9,1.2,'Iris-versicolor'], [6.0,2.7,5.1,1.6,'Iris-versicolor'],
    [5.4,3.0,4.5,1.5,'Iris-versicolor'], [6.0,3.4,4.5,1.6,'Iris-versicolor'],
    [6.7,3.1,4.7,1.5,'Iris-versicolor'], [6.3,2.3,4.4,1.3,'Iris-versicolor'],
    [5.6,3.0,4.1,1.3,'Iris-versicolor'], [5.5,2.5,4.0,1.3,'Iris-versicolor'],
    [5.5,2.6,4.4,1.2,'Iris-versicolor'], [6.1,3.0,4.6,1.4,'Iris-versicolor'],
    [5.8,2.6,4.0,1.2,'Iris-versicolor'], [5.0,2.3,3.3,1.0,'Iris-versicolor'],
    [5.6,2.7,4.2,1.3,'Iris-versicolor'], [5.7,3.0,4.2,1.2,'Iris-versicolor'],
    [5.7,2.9,4.2,1.3,'Iris-versicolor'], [6.2,2.9,4.3,1.3,'Iris-versicolor'],
    [5.1,2.5,3.0,1.1,'Iris-versicolor'], [5.7,2.8,4.1,1.3,'Iris-versicolor'],

    [6.3,3.3,6.0,2.5,'Iris-virginica'], [5.8,2.7,5.1,1.9,'Iris-virginica'],
    [7.1,3.0,5.9,2.1,'Iris-virginica'], [6.3,2.9,5.6,1.8,'Iris-virginica'],
    [6.5,3.0,5.8,2.2,'Iris-virginica'], [7.6,3.0,6.6,2.1,'Iris-virginica'],
    [4.9,2.5,4.5,1.7,'Iris-virginica'], [7.3,2.9,6.3,1.8,'Iris-virginica'],
    [6.7,2.5,5.8,1.8,'Iris-virginica'], [7.2,3.6,6.1,2.5,'Iris-virginica'],
    [6.5,3.2,5.1,2.0,'Iris-virginica'], [6.4,2.7,5.3,1.9,'Iris-virginica'],
    [6.8,3.0,5.5,2.1,'Iris-virginica'], [5.7,2.5,5.0,2.0,'Iris-virginica'],
    [5.8,2.8,5.1,2.4,'Iris-virginica'], [6.4,3.2,5.3,2.3,'Iris-virginica'],
    [6.5,3.0,5.5,1.8,'Iris-virginica'], [7.7,3.8,6.7,2.2,'Iris-virginica'],
    [7.7,2.6,6.9,2.3,'Iris-virginica'], [6.0,2.2,5.0,1.5,'Iris-virginica'],
    [6.9,3.2,5.7,2.3,'Iris-virginica'], [5.6,2.8,4.9,2.0,'Iris-virginica'],
    [7.7,2.8,6.7,2.0,'Iris-virginica'], [6.3,2.7,4.9,1.8,'Iris-virginica'],
    [6.7,3.3,5.7,2.1,'Iris-virginica'], [7.2,3.2,6.0,1.8,'Iris-virginica'],
    [6.2,2.8,4.8,1.8,'Iris-virginica'], [6.1,3.0,4.9,1.8,'Iris-virginica'],
    [6.4,2.8,5.6,2.1,'Iris-virginica'], [7.2,3.0,5.8,1.6,'Iris-virginica'],
    [7.4,2.8,6.1,1.9,'Iris-virginica'], [7.9,3.8,6.4,2.0,'Iris-virginica'],
    [6.4,2.8,5.6,2.2,'Iris-virginica'], [6.3,2.8,5.1,1.5,'Iris-virginica'],
    [6.1,2.6,5.6,1.4,'Iris-virginica'], [7.7,3.0,6.1,2.3,'Iris-virginica'],
    [6.3,3.4,5.6,2.4,'Iris-virginica'], [6.4,3.1,5.5,1.8,'Iris-virginica'],
    [6.0,3.0,4.8,1.8,'Iris-virginica'], [6.9,3.1,5.4,2.1,'Iris-virginica'],
    [6.7,3.1,5.6,2.4,'Iris-virginica'], [6.9,3.1,5.1,2.3,'Iris-virginica'],
    [5.8,2.7,5.1,1.9,'Iris-virginica'], [6.8,3.2,5.9,2.3,'Iris-virginica'],
    [6.7,3.3,5.7,2.5,'Iris-virginica'], [6.7,3.0,5.2,2.3,'Iris-virginica'],
    [6.3,2.5,5.0,1.9,'Iris-virginica'], [6.5,3.0,5.2,2.0,'Iris-virginica'],
    [6.2,3.4,5.4,2.3,'Iris-virginica'], [5.9,3.0,5.1,1.8,'Iris-virginica'],
])

X = data[:, :-1].astype(float)
y = data[:, -1]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# CELL 3 - Buat & compile model
# ============================================================
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# CELL 4 - Training model
# ============================================================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# ============================================================
# CELL 5 - Evaluasi & plot grafik
# ============================================================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Training History')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()


# ============================================================
# CELL 6 - Prediksi & confusion matrix
# ============================================================
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("Prediksi:", predicted_classes)
print("Label Asli:", y_test)

cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# ============================================================
# CELL 7 - Fungsi prediksi data baru
# ============================================================
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width  = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width  = float(input("Masukkan petal width: "))

    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()