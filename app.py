import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_option_menu import option_menu
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, classification_report

# prepocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

st.set_page_config(page_title="SKRIPSI-ISNAINI", page_icon='logo-utm.png')
@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

df = pd.read_csv('https://raw.githubusercontent.com/isnaini-ina/Skripsiina/refs/heads/main/dataset_framingham.csv')
df_imputasi = pd.read_excel('hasil_imputasi..xlsx')
df_normalisasi = pd.read_excel('hasil_normalisasi..xlsx')
df_hapusfitur = pd.read_excel('hasil_dropfitur...xlsx')
data_new = pd.read_excel('data_new.xlsx')

svm = joblib.load('model_efsvm/svm90.pkl')

with st.sidebar:
    selected = option_menu('',['Home', 'Research', 'Dataset', 'Preprocessing', 'Modelling', 'Implementation'], default_index=0)
if (selected == 'Home'):
    st.markdown("<h1 style='text-align: center; '>Klasifikasi terhadap Resiko Penyakit Jantung dengan Menggunakan Metode Entropy Fuzzy Support Vector Machine</h1>", unsafe_allow_html=True)
    url_logo = 'logo-utm.png'
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(url_logo, use_column_width='auto')
    with col3:
        st.write(' ')
    st.markdown("<h5 style='text-align: center; '>Diajukan Untuk Memenuhi Persyaratan Penyelesaian Studi Strata Satu (S1) dan Memperoleh Gelar Sarjana Komputer (S.Kom) di Universitas Trunojoyo Madura</h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>Disusun Oleh:</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>Isnaini</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; '>NPM : 200411100038</h6>", unsafe_allow_html=True)

    dp = [
        ("Dosen Pembimbing 1", "Dr. Bain Khusnul Khotimah, ST., M.Kom"),
        ("Dosen Pembimbing 2", "Dr. Eka Mala Sari Rochman, S.Kom. M.Kom")
    ]
    dp_table = pd.DataFrame(dp, columns=["Role", "Name"])
    st.table(dp_table)

    dpn = [
        ("Dosen Penguji 1", "Dwi Kuswanto, S.Pd., M.T."),
        ("Dosen Penguji 2", "Dr. Hermawan, ST., M.Kom."),
        ("Dosen Penguji 3", "Prof. Dr. Aeri Rachmad, ST., MT.")
    ]
    dpn_table = pd.DataFrame(dpn, columns=["Role", "Name"])
    st.table(dpn_table)

if (selected == 'Research'):
    st.markdown("<h1 style='text-align: center; '>Klasifikasi terhadap Resiko Penyakit Jantung dengan Menggunakan Metode Entropy Fuzzy Support Vector Machine</h1>", unsafe_allow_html=True)
    st.subheader("""Latar Belakang""")
    st.write("""Penelitian ini merupakan perbaikan dari metode sebelumnya yakni Fuzzy Entropy Support Machine yang ditulis oleh Faroh Ladayya, dkk (2017), di dalam penelitiaannya penggunaan fuzzy dalam SVM bertujuan untuk mengatasi ketidakseimbangan dengan memberikan nilai keanggotaan fuzzy pada setiap sampel data. Ini membantu mengurangi dampak sampel outlier atau data yang sangat tidak seimbang, sehingga meningkatkan akurasi klasifikasi. Namun, hasil penelitian menunjukkan bahwa meskipun ada peningkatan, metode FSVM masih mengalami bias terhadap kelas mayoritas ketika data sangat tidak seimbang.   
    Sehingga pada penelitian ini, dilakukan perbaikan dengan mengimplementasikan entropy fuzzy menggunakan algoritma dari Support Vector Machine (SVM). Dengan menambahkan entropy ke dalam keanggotaan fuzzy,dan oversampling sehingga model dapat lebih mempertimbangkan sampel dari kelas minoritas dan mengurangi dominasi kelas mayoritas, sehingga meningkatkan kinerja klasifikasi pada data yang tidak seimbang dan mengurangi bias, dibandingkan dengan menggunakan metode Support Vector Machine (SVM) biasa dan Fuzzy Support Vector Machine (FSVM) tanpa entropy.
    Data yang akan digunakan berasal dari repository kaggle yang merupakan data rekam medis pasien yang di diagnosis mengalami penyakit jantung.""")
    st.subheader("""Metode Usulan""")
    st.write("""Penelitian ini dilakukan untuk menangani imbalance pada data penyakit jantung menggunakan metode entropy fuzzy. Klasifikasi yang digunakan adalah Support Vector Machine. Pengujian performa terhadap klasifikasi dilakukan dengan melihat nilai statistik dari AUC, akurasi, sensitivity, specicifity. """)
    st.subheader("""Tujuan Penelitian""")
    st.write("""Berdasarkan latar belakang dan perumusan masalah yang diuraikan sebelumnya, tujuan dari penelitian ini sebagai berikut:
1.	Berapa presentase nilai akurasi ketika ditambahkan entropy fuzzy dalam menangani kasus data yang imbalance.
2.	Mengetahui perbandingan kinerja EFSVM dan SVM dalam mengklasifikasi penyakit jantung.
""")

if (selected == 'Dataset'):
    st.title("Dataset")
    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with dataset:
        dataset = pd.read_csv('https://raw.githubusercontent.com/isnaini-ina/Skripsiina/refs/heads/main/dataset_framingham.csv')
        # Optional: Display dataset
        if st.checkbox("Show Dataset"):
            st.write(dataset)
    with ket:
        st.write(
            "Pada penelitian ini dataset berasal dari situs kaggle.com. Data tersebut merupakan hasil studi kasus kardiovaskular yang sedang berlangsung pada penduduk kota Framingham, Massachusetts, Amerika Serikat. Studi Jantung di Kota Framingham sudah berdiri sejak tahun 1948 di bawah arahan National Heart, Lung, and Blood Institute (NHLBI) yang berfokus dalam mengidentifikasi faktor atau karakteristik umum yang berkontribusi terhadap penyakit kardiovaskular. Data ini dapat diakses secara publik melalui https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression")
        st.download_button(
            label="Download data",
            data='data_new.xlsx',
            file_name='data_new.xlsx',
            mime='text/xlsx',
        )
        st.write("""
            Keterangan Dataset :
        """)
        ket_data = pd.read_excel('data.xlsx')
        st.dataframe(ket_data)
        
if (selected == 'Preprocessing'):
    st.title("Preprocessing Data")
    mean_imputation, normalisasi_data, hapus_fitur= st.tabs(["Mean Imputation", "Normalisasi Data", "Hapus Fitur"])
    with mean_imputation:
        st.write("""Informasi missing value :""")
        mis = df.isnull().sum().reset_index()
        mis.columns = ['Fitur', 'Jumlah Missing Values']
        st.dataframe(mis, width=400)
        st.write("""Hasil Imputasi :""")
        st.dataframe(df_imputasi)
    with normalisasi_data:
        st.write("""Normalisasi data menggunakan Min-Max Normalization""")
        st.write("Rumus Min-Max :")
        st.latex(r"x_{\text{normalized}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}")
        st.write("""Hasil Normalisasi :""")
        st.dataframe(df_normalisasi)
    with hapus_fitur:
        korelasi_matrix = df_normalisasi.corr(method='pearson')
        fig = plt.figure(figsize=(15,10))
        sns.heatmap(korelasi_matrix, annot=True, cmap='coolwarm',vmin=-1, vmax=1)
        plt.title('Heatmap Matriks Korelasi')
        st.pyplot(fig)
        st.write("""Hasil Pengahapusan Fitur""")
        st.dataframe(df_hapusfitur)
        
if (selected == 'Modelling'):
    with st.form("Modelling"):
        st.subheader('Modelling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        svm = st.checkbox('Support Vector Machine (SVM)')
        efsvm = st.checkbox('Entropy Fuzzy + SVM (K=3, K=5, K=7)')
        efsvm_90 = st.checkbox('Split Data (90:10)')
        efsvm_80 = st.checkbox('Split Data (80:20)')
        efsvm_70 = st.checkbox('Split Data (70:30)')
        submitted = st.form_submit_button("Submit")

        # st.dataframe(data_new)
        X = data_new.drop('TenYearCHD', axis=1)  # Menghapus kolom target ('TenYearCHD') dari fitur
        y = data_new['TenYearCHD']  # Menetapkan kolom target 'TenYearCHD' sebagai y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=42)
        
        svm90 = joblib.load('model_efsvm/svm90.pkl')
        svm80 = joblib.load('model_efsvm/svm80.pkl')
        svm70 = joblib.load('model_efsvm/svm70.pkl')
        # Prediksi dan probabilitas untuk data uji
        y_pred_svm = svm90.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_svm)
        cm = confusion_matrix(y_test, y_pred_svm)

        y_pred_svm2 = svm80.predict(X_test2)
        accuracy2 = accuracy_score(y_test2, y_pred_svm2)
        cm2 = confusion_matrix(y_test2, y_pred_svm2)

        y_pred_svm3 = svm70.predict(X_test3)
        accuracy3 = accuracy_score(y_test3, y_pred_svm3)
        cm3 = confusion_matrix(y_test3, y_pred_svm3)
        
        if submitted:
            if svm:
                # Visualisasi dan evaluasi model SVM
                st.write('SVM Keseluruhan')
                st.image('SVM_keseluruhan.png')
                st.write('Confussion Matrik Tertinggi')
                st.image('confussion_matrik_SVM.png')
                st.write('Classification Report Tertinggi')
                st.image('classificationreport_svm.png')
            if efsvm:
                # Visualisasi dan evaluasi model EFSVM dengan K=3,5,7
                st.write('EFSVM dengan K=7')
                st.image('efsvm7.png')
                st.write('EFSVM dengan K=5')
                st.image('efsvm5.png')
                st.write('EFSVM dengan K=3')
                st.image('efsvm3.png')
            if efsvm_90:
                st.write('Accuracy: {0:0.2f}'. format(accuracy))
                st.image('classification_skenario3.png')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            if efsvm_80:
                st.write('Accuracy: {0:0.2f}'. format(accuracy2))
                st.image('classifcationreport2.png')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            if efsvm_70:
                st.write('Accuracy: {0:0.2f}'. format(accuracy3))
                st.image('classificationreport3.png')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                st.write('Diagram Perbandingan Keseluruhan EFSVM')
                st.image('efsvm_keseluruhan.png')
                
if (selected == "Implementation"):
     with st.form("my_form"):
        st.subheader("Implementation")
        male = st.selectbox('Male', options=[0,1])
        age = st.number_input('Age (Usia Pasien)', min_value=0, max_value=1000)
        currentSmoker = st.selectbox('currentSmoker (Perokok)', options=[0,1]) 
        cigsPerDay = st.number_input('cigsPerDay (Jumlah Rokok Perhari)',  min_value=0, max_value=1000) 
        BPMeds = st.selectbox('BPMeds (Menjalani Pengobatan Tekanan Darah )', options=[0,1]) 
        prevalentStroke = st.selectbox('prevalentStroke (Mengalami Penyakit Stroke )', options=[0,1]) 
        prevalentHyp = st.selectbox('prevalentHyp (Mengalami Hypertensi)', options=[0,1]) 
        diabetes = st.selectbox('diabetes (Menderita Diabetes)', options=[0,1])  
        totChol = st.number_input('TotChol (Jumlah Kadar Kolestrol)',  min_value=0)
        sysBP = st.number_input('SysBP (Tekanan Darah Sistolik)',  min_value=0)
        diaBP = st.number_input('DiaBP (Tekanan Darah Diastolik)',  min_value=0)
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=0.0)
        heartRate = st.number_input('heartRate (Denyut Jantung)', min_value=0.0)
        glucose = st.number_input('Glukosa (Kadar Glukosa)',  min_value=0)
         
        prediksi = st.form_submit_button("Predict")
        if prediksi:
            input = {
                'male': [male],
                'age': [age],
                'currentSmoker': [currentSmoker],
                'cigsPerDay': [cigsPerDay],
                'BPMeds': [BPMeds],
                'prevalentStroke': [prevalentStroke],
                'prevalentHyp': [prevalentHyp],
                'diabetes': [diabetes],
                'totChol': [totChol],
                'sysBP': [sysBP],
                'diaBP': [diaBP],
                'BMI': [bmi],
                'heartRate': [heartRate],
                'glucose':[glucose]
            }
            input_data = pd.DataFrame(input)
            input_data = input_data.astype(float)
            scaler = MinMaxScaler()
            input_data_scaled = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)
            prediction = svm.predict(input_data_scaled)

            st.subheader('Prediction Results')
            
            if prediction == 1:
                st.error("Hasil Prediksi: Penyakit Jantung Terdiagnosis (1)")
            else:
                st.success("Hasil Prediksi: Tidak Terdiagnosis Penyakit Jantung (0)")
        
