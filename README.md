## Jaringan Syaraf Tiruan untuk Klasifikasi Bunga Iris
Program ini merupakan implementasi Jaringan Syaraf Tiruan menggunakan TensorFlow dan Keras untuk mengklasifikasikan jenis bunga Iris. Dataset yang digunakan berisi data morfologi bunga Iris, yaitu panjang sepal, lebar sepal, panjang petal, dan lebar petal.
Jenis bunga Iris yang diklasifikasikan dalam program ini adalah:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Library yang Digunakan
Program ini menggunakan beberapa library, yaitu:
1. **TensorFlow**  
   Digunakan untuk membangun dan melatih model jaringan syaraf tiruan.
2. **Keras**  
   Digunakan untuk membuat arsitektur model neural network secara sederhana melalui `Sequential`, `Input`, dan `Dense`.
3. **NumPy**  
   Digunakan untuk menyimpan dan mengolah dataset dalam bentuk array.
4. **Pandas**  
   Digunakan untuk mengubah riwayat proses training menjadi DataFrame agar dapat divisualisasikan dalam bentuk grafik.
5. **Scikit-learn**  
   Digunakan untuk proses preprocessing dan evaluasi, seperti mengubah label menjadi angka, membagi dataset, dan membuat confusion matrix.
6. **Matplotlib**  
   Digunakan untuk menampilkan grafik hasil training dan visualisasi confusion matrix.
7. **Seaborn**  
   Digunakan untuk membuat tampilan confusion matrix menjadi lebih rapi dan mudah dibaca.

## Penjelasan Kode
### 1. Import Library
Pada bagian awal program dilakukan import library yang dibutuhkan.
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

Library tersebut digunakan untuk membuat model, mengolah data, membagi dataset, mengevaluasi hasil prediksi, dan menampilkan visualisasi.

### 2. Load Dataset
Dataset dimasukkan secara langsung ke dalam program menggunakan `np.array`.
data = np.array([...])
Setiap data terdiri dari 5 nilai, yaitu:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width
5. Label jenis bunga Iris

Contoh data:
[5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
Empat nilai pertama digunakan sebagai fitur, sedangkan nilai terakhir digunakan sebagai label kelas.

### 3. Memisahkan Fitur dan Label
Dataset kemudian dipisahkan menjadi dua bagian, yaitu `X` dan `y`.
X = data[:, :-1].astype(float)
y = data[:, -1]

`X` berisi fitur bunga, yaitu ukuran sepal dan petal.
`y` berisi label atau jenis bunga Iris.

### 4. Encode Label
Label awalnya berbentuk teks, seperti `Iris-setosa`, `Iris-versicolor`, dan `Iris-virginica`. Agar dapat diproses oleh model, label tersebut diubah menjadi angka menggunakan `LabelEncoder`.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
Hasil encoding:
* Iris-setosa menjadi 0
* Iris-versicolor menjadi 1
* Iris-virginica menjadi 2

### 5. Membagi Dataset
Dataset dibagi menjadi data training dan data testing menggunakan `train_test_split`.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
Pembagian data dilakukan dengan rasio:
* 80% data training
* 20% data testing
Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk menguji performa model.

### 6. Membuat Model Jaringan Syaraf Tiruan
Model dibuat menggunakan `Sequential`.
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

Arsitektur model terdiri dari:
1. Input layer
   Menerima 4 fitur input, yaitu sepal length, sepal width, petal length, dan petal width.
2. Hidden layer pertama
   Menggunakan 1000 neuron dengan aktivasi ReLU.
3. Hidden layer kedua
   Menggunakan 500 neuron dengan aktivasi ReLU.
4. Hidden layer ketiga
   Menggunakan 300 neuron dengan aktivasi ReLU.
5. Output layer
   Menggunakan 3 neuron karena terdapat 3 kelas bunga Iris. Aktivasi yang digunakan adalah softmax.
Fungsi aktivasi ReLU digunakan pada hidden layer agar model dapat mempelajari pola yang lebih kompleks. Softmax digunakan pada output layer karena kasus ini merupakan klasifikasi multikelas.

### 7. Menampilkan Struktur Model
model.summary()
Perintah ini digunakan untuk menampilkan ringkasan arsitektur model, seperti jumlah layer, bentuk output, dan jumlah parameter yang dilatih.

### 8. Compile Model
Sebelum proses training, model dikompilasi terlebih dahulu.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

Penjelasan:
* `optimizer='adam'` digunakan untuk memperbarui bobot model selama proses training.
* `loss='sparse_categorical_crossentropy'` digunakan karena label berbentuk angka dan klasifikasi memiliki lebih dari dua kelas.
* `metrics=['accuracy']` digunakan untuk mengukur tingkat akurasi model.

### 9. Training Model
Model dilatih menggunakan data training.
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

Penjelasan:
* `epochs=50` berarti model dilatih sebanyak 50 kali putaran.
* `batch_size=32` berarti data diproses sebanyak 32 data dalam satu batch.
* `validation_data=(X_test, y_test)` digunakan untuk mengevaluasi model pada data testing selama proses training.
Hasil training disimpan dalam variabel `history`.

### 10. Evaluasi Model
Setelah training selesai, model dievaluasi menggunakan data testing.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
Evaluasi menghasilkan nilai:
* Loss, yaitu tingkat kesalahan model.
* Accuracy, yaitu tingkat ketepatan model dalam melakukan prediksi.

Semakin kecil nilai loss dan semakin besar nilai accuracy, maka performa model semakin baik.

### 11. Visualisasi Training
Riwayat training divisualisasikan dalam bentuk grafik.
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Training History')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()

Grafik ini menampilkan perkembangan nilai loss dan accuracy selama proses training. Dari grafik tersebut dapat dilihat apakah model mengalami peningkatan performa dari epoch ke epoch.

### 12. Prediksi Data Testing
Model digunakan untuk memprediksi data testing.
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

`model.predict(X_test)` menghasilkan nilai probabilitas dari setiap kelas.
`argmax(axis=1)` digunakan untuk mengambil kelas dengan probabilitas tertinggi sebagai hasil prediksi akhir.

### 13. Menampilkan Hasil Prediksi
print("Prediksi:", predicted_classes)
print("Label Asli:", y_test)

Bagian ini menampilkan hasil prediksi model dan membandingkannya dengan label asli dari data testing.

### 14. Confusion Matrix
Confusion matrix digunakan untuk melihat hasil klasifikasi model secara lebih detail.
cm = confusion_matrix(y_test, predicted_classes)
```
Confusion matrix menunjukkan jumlah prediksi yang benar dan salah pada setiap kelas.
Visualisasi confusion matrix dibuat menggunakan Seaborn.

]]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

Keterangan:

* Sumbu X menunjukkan hasil prediksi model.
* Sumbu Y menunjukkan label asli.
* Angka pada kotak menunjukkan jumlah data pada masing-masing kombinasi prediksi dan label asli.

---

### 15. Fungsi Prediksi Data Baru

Program juga menyediakan fungsi untuk memprediksi data baru yang dimasukkan oleh pengguna.

```python
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
```

Pengguna diminta memasukkan empat nilai fitur bunga, yaitu sepal length, sepal width, petal length, dan petal width.

Data tersebut kemudian diprediksi oleh model. Hasil prediksi yang awalnya berupa angka dikembalikan lagi menjadi nama kelas menggunakan `inverse_transform`.

Contoh output:

```text
Prediksi kelas: Iris-setosa
```

---

## Cara Menjalankan Program

Install library yang dibutuhkan:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

Jalankan program:

```bash
python modul7.py
```

Setelah program dijalankan, sistem akan melakukan training model, menampilkan hasil evaluasi, menampilkan confusion matrix, dan meminta input data baru untuk diprediksi.

---

