# Menemukan perbandingan performa model algoritma logistic regression dengan algoritma naive bayes menggunakan metrik-metrik evaluasi seperti akurasi, presisi, recall, dan f1-score
# Gunakan sklearn untuk menghitung metrik-metrik tersebut.

# Import library
from sklearn.metrics import accuracy_score

# Hitung akurasi model logistic regression
acc_logreg = accuracy_score(y_test, y_pred_logreg)

# Hitung akurasi model naive bayes
acc_nb = accuracy_score(y_test, y_pred_nb)

# Bandingkan akurasi kedua model
print(f"Akurasi model logistic regression: {acc_logreg}")
print(f"Akurasi model naive bayes: {acc_nb}")