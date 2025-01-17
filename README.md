# Skin Disease Detection using LVQ and Streamlit

## Deskripsi Singkat
Proyek ini bertujuan untuk mengklasifikasikan gambar penyakit kulit yang dimasukkan oleh pengguna. Website ini dirancang menggunakan Streamlit dan dirancang khusus untuk memenuhi kebutuhan klien freelance Gueanakampus. Dataset gambar yang digunakan bersumber dari klien, dengan proses training dan deployment yang dilakukan secara efisien.

Tahapan proyek meliputi preprocessing dataset, ekstraksi fitur, pelatihan model, dan deployment website. Model dikembangkan menggunakan algoritma LVQ (Learning Vector Quantization) dengan akurasi lebih dari 90%. Website ini memanfaatkan Python dan berbagai pustaka pendukung untuk analisis gambar dan klasifikasi.

## Screenshot dan Visualisasi

1. **Preprocessing Dataset**

![Image](https://github.com/user-attachments/assets/81dcbbeb-cd85-4a5c-9f13-394e4de8d369)

2. **Feature Extraction**

![Image](https://github.com/user-attachments/assets/2c42993d-db68-4bb6-b6f9-74b28df505c8)

3. **Rata-rata Histogram LBP per-class**

![Image](https://github.com/user-attachments/assets/399c4a0b-44d4-4705-b913-0f8ef211022e)

4. **Confusion Matrix sebagai Evaluasi Model**

![Image](https://github.com/user-attachments/assets/d0d2a2ec-50f2-441f-b849-01be38d6bfe4)

5. **Hasil Testing terhadap Model**

![Image](https://github.com/user-attachments/assets/358fad2e-48d3-4865-8e3f-f903384b8af4)

## Teknologi yang Digunakan
- **Python Libraries**:
  - OpenCV
  - Scikit-learn
  - Scikit-image
  - Streamlit
- **Framework**:
  - Streamlit untuk website

## Cara Install dan Pemakaian
Ikuti langkah-langkah berikut untuk menginstall dan menjalankan proyek ini:

### Instalasi
1. **Persiapkan Lingkungan Python**:
   - Pastikan Python 3.8 atau versi lebih baru telah terinstal.
   - Install pustaka yang diperlukan dengan perintah berikut:
     ```bash
     pip install --upgrade scikit-learn opencv-python-headless scikit-image streamlit
     ```

2. **Clone Repository**:
   - Clone repository proyek ini:
     ```bash
     git clone <repository-link>
     ```

3. **Persiapkan Model**:
   - Pastikan file model dalam format `.pkl` telah berada di folder proyek.
   - Dataset yang telah diproses tersedia jika diperlukan.

### Menjalankan Website Streamlit
1. Buka terminal atau command prompt, arahkan ke direktori proyek.
2. Jalankan perintah berikut untuk memulai aplikasi:
   ```bash
   streamlit run app.py
   ```
3. Buka browser dan akses aplikasi di `http://localhost:8501` atau URL yang diberikan oleh Streamlit.

### Penggunaan
- **Upload Gambar**: Pengguna dapat mengunggah gambar penyakit kulit melalui antarmuka web.
- **Klasifikasi**: Gambar yang diunggah akan melalui preprocessing dan klasifikasi oleh model LVQ.
- **Hasil Prediksi**: Website akan menampilkan hasil klasifikasi penyakit kulit berdasarkan gambar yang dimasukkan.

## Penjelasan Proyek
Proyek ini terdiri dari beberapa tahapan utama:

1. **Preprocessing Dataset**:
   - Grayscaling: Mengubah gambar menjadi skala abu-abu.
   - Resizing: Menyesuaikan ukuran gambar agar seragam.
   - Min-Max Scaling: Normalisasi nilai piksel.
   - Reshaping: Mengubah dimensi gambar agar sesuai untuk ekstraksi fitur.

2. **Feature Extraction**:
   - Menggunakan Local Binary Pattern (LBP) dengan parameter:
     - **Number of Points**: 32
     - **Radius**: 4

3. **Training Model**:
   - Splitting data menjadi data latih dan data uji.
   - Melakukan tuning hyperparameter pada model LVQ.
   - Evaluasi model menggunakan metrik seperti akurasi, precision, recall, dan F1-score.

4. **Deployment**:
   - Model disimpan dalam format `.pkl` menggunakan Pickle.
   - Website dibangun dengan Streamlit untuk memberikan antarmuka yang interaktif kepada pengguna.


