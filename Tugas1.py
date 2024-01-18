# Mengganti algoritma logistic regression dengan algoritma naive bayes menggunakan sklearn
# Algoritma naive bayes menghitung probabilitas posterior dari kelas yang diberikan fitur dengan menggunakan teorema Bayes dan asumsi independensi

# Import library
from sklearn.naive_bayes import GaussianNB

# Buat model naive bayes
model = GaussianNB()

# Latih model dengan data training
model.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = model.predict(X_test)

