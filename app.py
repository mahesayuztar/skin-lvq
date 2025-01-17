import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
import seaborn as sns
import pickle 

minmax_dibaca = pd.read_excel('lbp_minmax.xlsx')
min = minmax_dibaca['Min']
min.index = [f'LBP_{i}' for i in range(minmax_dibaca.shape[0])]
max = minmax_dibaca['Max']
max.index = [f'LBP_{i}' for i in range(minmax_dibaca.shape[0])]

def save_numpy_to_excel(array, path, sheet_name='Sheet1', header=None, index=False):
    """
    Menyimpan array NumPy ke file Excel.

    Parameters:
    - array (np.ndarray): Array NumPy yang akan disimpan.
    - path (str): Path lengkap untuk menyimpan file Excel, termasuk nama file dan ekstensi (.xlsx).
    - sheet_name (str): Nama sheet dalam file Excel. Default 'Sheet1'.
    - header (list or None): Daftar header kolom. Jika None, tidak ada header.
    - index (bool): Jika True, tambahkan indeks ke file Excel. Default False.
    
    Returns:
    - None
    """
    print("masuk sini kok")
    try:
        # Konversi NumPy array ke DataFrame
        df = pd.DataFrame(array)

        # Tambahkan header jika diberikan
        if header:
            df.columns = header
            
        # Simpan DataFrame ke file Excel
        df.to_excel(path, index=index, sheet_name=sheet_name)
        print(f"Array berhasil disimpan ke file Excel: {path}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

def scale_numeric_columns(df, min_vals = min, max_vals = max):
    """
    Scale numeric columns in a DataFrame using Min-Max Scaling.
    
    Parameters:
    - df (pd.DataFrame): DataFrame yang akan di-scaling
    - min_vals (dict): Dictionary dengan format {column_name: min_value}
    - max_vals (dict): Dictionary dengan format {column_name: max_value}
    
    Returns:
    - pd.DataFrame: DataFrame dengan kolom numerik yang telah di-scale
    """
    scaled_df = df.copy()  # Salin DataFrame asli untuk di-scale
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in min_vals and col in max_vals:  # Pastikan kolom ada di min/max
            min_val = min_vals[col]
            max_val = max_vals[col]
            scaled_df[col] = (df[col] - min_val) / (max_val - min_val)  # Scaling
        else:
            print(f"Skipping column '{col}' as it lacks min or max values.")
    
    return scaled_df

label_map = {
    0: "Campak",
    1: "Herpes",
    2: "Cacar Air",
    3: "Cacar Monyet"
}

def predict_single_image(image, lvq, target_size=(500, 500), P=32, R=4):
    """
    image_path: path ke gambar yang akan diuji
    lvq: model LVQ yang telah dilatih
    target_size: ukuran gambar yang di-resize
    P, R: Parameter untuk LBP
    """
    # Load dan preprocess gambar
    image = image.convert('RGB')
    image = np.array(image)
    image= image[..., ::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(image, (500,500)) / 255.0
    
    # Ekstraksi fitur LBP
    img_array = (img_array * 255).astype(np.uint8)  # Konversi ke uint8
    lbp = local_binary_pattern(img_array, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalisasi histogram
    img_array = hist  # Gunakan histogram sebagai input model

    # Debugging: Periksa dimensi fitur
    # print(f"LBP histogram for single image: {img_array}")
    img_array = img_array.reshape(1,-1)
    img_df = pd.DataFrame(img_array, columns=[f'LBP_{i}' for i in range(minmax_dibaca.shape[0])])
    img_df = scale_numeric_columns(img_df)
    img_array = np.array(img_df)

    # Prediksi dengan model
    prediction = lvq.predict([img_array])  # Pass the LBP features for prediction

    # Debugging: Periksa hasil prediksi
    # print(f"Prediction index: {prediction[0]})")

    # # Konversi indeks prediksi ke label
    predicted_label = label_map.get(prediction[0], "Unknown")
    print(f"Predicted label: {predicted_label}")
    # predicted_label = "dimatikan"
    return predicted_label

def preprocess_image(image):
    """
    Preprocess the uploaded image to prepare it for model inference.
    """
    
    image = image.convert('RGB')  # Ensure RGB format
    image_array = np.array(image)
    image_array = image_array[..., ::-1]
    
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    save_numpy_to_excel(image_gray, 'image_gray_web.xlsx', sheet_name='Sheet1')
    image_resized = cv2.resize(image_gray, (500, 500)) / 255.0

    # Extract LBP features
    radius = 4
    n_points = 40
    image_process = (image_resized * 255).astype(np.uint8)
    
    lbp = local_binary_pattern(image_process, P=n_points, R=radius, method="uniform")
    
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize histogram
    hist = hist.flatten()
    return hist  # Reshape for model compatibility

class LVQ:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.prototypes = None
        self.labels = None

    def fit(self, X, y, n_prototypes_per_class=50):
        """
        X: array data input (n_samples, n_features)
        y: array label (n_samples,)
        n_prototypes_per_class: jumlah prototype per kelas
        """
        np.random.seed(42)

        # Identifikasi jumlah kelas unik
        classes = np.unique(y)
        self.prototypes = []
        self.labels = []

        # Inisialisasi vektor prototipe untuk setiap kelas
        for cls in classes:
            class_samples = X[y == cls]
            for _ in range(n_prototypes_per_class):
                idx = np.random.choice(len(class_samples))
                self.prototypes.append(class_samples[idx])
                self.labels.append(cls)

        self.prototypes = np.array(self.prototypes)

        # Proses pelatihan
        for epoch in range(self.max_epochs):
            for xi, yi in zip(X, y):
                # Hitung jarak antara data dan prototipe
                distances = np.linalg.norm(self.prototypes - xi, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_label = self.labels[nearest_idx]

                # Update prototipe jika label cocok atau tidak
  # Update prototipe
                if yi == nearest_label:
                    self.prototypes[nearest_idx] += self.learning_rate * (xi - self.prototypes[nearest_idx])
                else:
                    self.prototypes[nearest_idx] -= self.learning_rate * (xi - self.prototypes[nearest_idx])



            # Update learning rate (opsional: bisa dikurangi setiap epoch)
            self.learning_rate *= 0.99

    def predict(self, X):
        """
        X: array data input (n_samples, n_features)
        """
        y_pred = []
        for xi in X:
            # Reshape xi to (1, n_features) for broadcasting
            xi = xi.reshape(1, -1)  # This will reshape xi to (1, 10)
            distances = np.linalg.norm(self.prototypes - xi, axis=1)
            # st.write(f"Distances: {distances}")
            nearest_idx = np.argmin(distances)
            # st.write(f"Nearest index: {nearest_idx}")
            y_pred.append(self.labels[nearest_idx])
        return np.array(y_pred)


def load_model():
    """
    Load the trained LVQ model.
    """
    try:
        with open('model_lvq_baru.pkl', 'rb') as model_file:
            lvq = pickle.load(model_file)
        return lvq
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'lvq.pkl' is available.")
        return None

def main():
    st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")
    st.title("🩺 Deteksi Penyakit Kulit")
    st.markdown("Website ini memungkinkan Anda untuk mengunggah gambar dan mendeteksi kelas penyakit kulit menggunakan model LVQ.")

    # File uploader
    uploaded_file = st.file_uploader("Pilih gambar kulit...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
  
        # Preprocess the image
        # st.image(processed_image, caption="Gambar preprocessed", use_container_width=True)

        # Load the model
        lvq = load_model()
        
        if lvq is not None:
            # Predict the class
            # st.write(f"Prediction index: {len(lvq.prototypes)}")

            prediction = predict_single_image(image, lvq)
            # st.write(f"Prediction index: {prediction[0]} (type: {type(prediction[0])})")

            # Display the prediction
            st.success(f"Model mendeteksi bahwa gambar ini adalah: {prediction}")

if __name__ == "__main__":
    main()
