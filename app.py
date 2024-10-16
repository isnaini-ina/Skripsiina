import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# prepocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

st.set_page_config(page_title="SKRIPSI-ina", page_icon='logo-utm.png')
@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

df = pd.read_csv('https://raw.githubusercontent.com/isnaini-ina/Skripsiina/refs/heads/main/framingham.csv')
df_imputasi = pd.read_excel('hasil_imputasi.xlsx')
df_normalisasi = pd.read_excel('hasil_normalisasi.xlsx')
df_oversampling = pd.read_excel('hasil_oversampling.xlsx')
df_hapusfitur = pd.read_excel('setelah_hapusfitur.xlsx')
df_IG = pd.read_excel('urutan_IG.xlsx')
df_topfitur = pd.read_excel('hasil_topfitur.xlsx')

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
        ("Dosen Pembimbing 1", "Dr. Bain Khusnul Khotimah, S.T., M.Kom"),
        ("Dosen Pembimbing 2", "Dr. Eka Mala Sari Rochman, S.Kom. M.Kom")
    ]
    dp_table = pd.DataFrame(dp, columns=["Role", "Name"])
    st.table(dp_table)

    dpn = [
        ("Dosen Penguji 1", "Dwi Kuswanto, S.Pd., M.T."),
        ("Dosen Penguji 2", "Hermawan, ST., M.Kom."),
        ("Dosen Penguji 3", "Prof. Aeri Rachmad, S.T., M.T.")
    ]
    dpn_table = pd.DataFrame(dpn, columns=["Role", "Name"])
    st.table(dpn_table)

if (selected == 'Research'):
    st.markdown("<h1 style='text-align: center; '>Klasifikasi terhadap Resiko Penyakit Jantung dengan Menggunakan Metode Entropy Fuzzy Support Vector Machine</h1>", unsafe_allow_html=True)
    st.subheader("""Latar Belakang""")
    st.write("""Penelitian ini merupakan perbaikan dari metode sebelumnya yakni Fuzzy Entropy Support Machine yang ditulis oleh Faroh Ladayya, dkk (2017), di dalam penelitiaannya penggunaan fuzzy dalam SVM bertujuan untuk mengatasi ketidakseimbangan dengan memberikan nilai keanggotaan fuzzy pada setiap sampel data. Ini membantu mengurangi dampak sampel outlier atau data yang sangat tidak seimbang, sehingga meningkatkan akurasi klasifikasi. Namun, hasil penelitian menunjukkan bahwa meskipun ada peningkatan, metode FSVM masih mengalami bias terhadap kelas mayoritas ketika data sangat tidak seimbang[5].   
    Sehingga pada penelitian ini, dilakukan perbaikan dengan mengimplementasikan entropy fuzzy menggunakan algoritma dari Support Vector Machine (SVM). Dengan menambahkan entropy ke dalam keanggotaan fuzzy,dan oversampling sehingga model dapat lebih mempertimbangkan sampel dari kelas minoritas dan mengurangi dominasi kelas mayoritas, sehingga meningkatkan kinerja klasifikasi pada data yang tidak seimbang dan mengurangi bias, dibandingkan dengan menggunakan metode Support Vector Machine (SVM) biasa dan Fuzzy Support Vector Machine (FSVM) tanpa entropy.
    Data yang akan digunakan berasal dari repository kaggle yang merupakan data rekam medis pasien yang di diagnosis mengalami penyakit jantung. Dari hasil penelitian ini 
     """)
    st.subheader("""Metode Usulan""")
    st.write("""Penelitian ini dilakukan untuk menangani imbalance pada data penyakit jantung menggunakan metode entropy fuzzy, dan oversampling. Klasifikasi yang digunakan adalah Support Vector Machine. Pengujian performa terhadap klasifikasi dilakukan dengan melihat nilai statistik dari AUC, akurasi, sensitivity, specicifity. """)
    st.subheader("""Tujuan Penelitian""")
    st.write("""Berdasarkan latar belakang dan perumusan masalah yang diuraikan sebelumnya, tujuan dari penelitian ini sebagai berikut:
1.	Berapa presentase nilai akurasi ketika ditambahkan entropy fuzzy dalam menangani kasus data yang imbalance.
2.	Mengetahui perbandingan kinerja EFSVM dan SVM dalam mengklasifikasi penyakit jantung.
""")

if (selected == 'Dataset'):
    st.title("Dataset")
    dataset, ket = st.tabs(['Dataset', 'Ket Dataset'])
    with dataset:
        dataset = pd.read_csv('https://raw.githubusercontent.com/isnaini-ina/Skripsiina/refs/heads/main/framingham.csv')
        # Optional: Display dataset
        if st.checkbox("Show Dataset"):
            st.write(dataset)
    with ket:
        st.write(
            "Pada penelitian ini dataset berasal dari situs kaggle.com. Data tersebut merupakan hasil studi kasus kardiovaskular yang sedang berlangsung pada penduduk kota Framingham, Massachusetts, Amerika Serikat. Studi Jantung di Kota Framingham sudah berdiri sejak tahun 1948 di bawah arahan National Heart, Lung, and Blood Institute (NHLBI) yang berfokus dalam mengidentifikasi faktor atau karakteristik umum yang berkontribusi terhadap penyakit kardiovaskular. Data ini dapat diakses secara publik melalui https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression")
        st.download_button(
            label="Download data",
            data='data.xlsx',
            file_name='data.xlsx',
            mime='text/xlsx',
        )
        st.write("""
            Keterangan Dataset :
        """)
        ket_data = pd.read_excel('data.xlsx')
        st.dataframe(ket_data)
        
if (selected == 'Preprocessing'):
    st.title("Preprocessing Data")
    mean_imputation, normalisasi_data, oversampling, hapus_fitur, seleksi_fitur= st.tabs(["Mean Imputation", "Normalisasi Data", "Oversampling", "Hapus Fitur", "Seleksi Fitur"])
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
    with oversampling:
        jumlah_sampel = df_oversampling['TenYearCHD'].value_counts()
        colors = ['red' if label == 0 else 'blue' for label in jumlah_sampel.index]
        plt.bar(jumlah_sampel.index, jumlah_sampel.values, color=colors)
        plt.title('Perbandingan Jumlah Data Negatif dan Positif Setelah Oversampling', fontsize=10)
        plt.xlabel('Kelas TenYearCHD', fontsize=10)
        plt.ylabel('Jumlah Data', fontsize=10)
        plt.xticks(ticks=[0, 1], labels=['Negatif (0)', 'Positif (1)']) 
        st.pyplot(plt)
        proporsi = df_oversampling['TenYearCHD'].value_counts()
        plt.figure(figsize=(4, 3))
        proporsi.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'blue'], labels=['Negatif', 'Positif'])
        plt.title('Proporsi Data TenYearCHD Setelah Oversampling', fontsize=7)
        plt.ylabel('') 
        st.pyplot(plt)
    with hapus_fitur:
        korelasi_matrix = df_oversampling.corr(method='pearson')
        fig = plt.figure(figsize=(15,10))
        sns.heatmap(korelasi_matrix, annot=True, cmap='coolwarm',vmin=-1, vmax=1)
        plt.title('Heatmap Matriks Korelasi')
        st.pyplot(fig)
        st.write("""Hasil Pengahapusan Fitur""")
        st.dataframe(df_hapusfitur)
    with seleksi_fitur:
        st.write("""Urutan Top Fitur""")
        st.dataframe(df_IG)
        plt.figure(figsize=(10, 6))
        plt.barh(df_IG['Feature'], df_IG['Information Gain'], color='skyblue')
        plt.xlabel('Information Gain')
        plt.title('Information Gain dari Setiap Fitur')
        plt.gca().invert_yaxis() 
        st.pyplot(plt)
        st.write("""Hasil Seleksi Fitur""")
        st.dataframe(df_topfitur)

if (selected == 'Modelling'):
    with st.form("Modelling"):
        st.subheader('Modelling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        svm = st.checkbox('Support Vector Machine (SVM)')
        efsvm = st.checkbox('Entropy Fuzzy + Support Vector Machine')
        efsvm_k3 = st.checkbox('Entropy Fuzzy + Support Vector Machine + Oversampling + Split Data (70:30)')
        efsvm_k5 = st.checkbox('Entropy Fuzzy + Support Vector Machine + Oversampling + Split Data (80:30)')
        efsvm_k7 = st.checkbox('Entropy Fuzzy + Support Vector Machine + Oversampling + Split Data (90:10)')
        submitted = st.form_submit_button("Submit")

if (selected == "Implementation"):
     with st.form("my_form"):
        st.subheader("Implementation")
        age = st.number_input('Age (Usia Pasien)', min_value=0, max_value=100, value=50)
        bmi = st.number_input('BMI (Indeks Massa Tubuh)', min_value=0.0, value=25.0)
        sysBP = st.number_input('SysBP (Tekanan Darah Sistolik)',  min_value=0, max_value=100, value=50)
        diaBP = st.number_input('DiaBP (Tekanan Darah Diastolik)',  min_value=0, max_value=100, value=50)
        totChol = st.number_input('TotChol (Jumlah Kadar Kolestrol)',  min_value=0, max_value=100, value=50)
        glucose = st.number_input('Glukosa (Kadar Glukosa)',  min_value=0, max_value=100, value=50)
        prevalentHyp = st.selectbox('PrevalentHyp (Mengalami Hipertensi(1)/tidak(0))', options=[0, 1])
        prediksi = st.form_submit_button("Predict")
        
