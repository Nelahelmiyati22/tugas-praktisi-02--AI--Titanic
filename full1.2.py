# Import library yang diperlukan
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset Titanic
data = pd.read_csv('titanic.csv')

# Preprocessing data
# ...

# Split dataset menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest
rf_model = RandomForestClassifier()

# Latih model Random Forest
rf_model.fit(X_train, y_train)

# Prediksi dengan model Random Forest
rf_predictions = rf_model.predict(X_test)

# Hitung akurasi model Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Inisialisasi model Naive Bayes
nb_model = GaussianNB()

# Latih model Naive Bayes
nb_model.fit(X_train, y_train)

# Prediksi dengan model Naive Bayes
nb_predictions = nb_model.predict(X_test)

# Hitung akurasi model Naive Bayes
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Bandingkan performa kedua model
print("Performa Model Random Forest: ", rf_accuracy)
print("Performa Model Naive Bayes: ", nb_accuracy)