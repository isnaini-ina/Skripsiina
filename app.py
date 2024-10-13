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
    st.markdown("<h6 style='text-align: center; '>NPM:200411100038</h6>", unsafe_allow_html=True)

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
            "Pada penelitian ini dataset berasal dari situs kaggle.com. Data tersebut merupakan hasil studi kasus kardiovaskular yang sedang berlangsung pada penduduk kota Framingham, Massachusetts, Amerika Serikat. Studi Jantung di Kota Framingham sudah berdiri sejak tahun 1948 di bawah arahan National Heart, Lung, and Blood Institute (NHLBI) yang berfokus dalam mengidentifikasi faktor atau karakteristik umum yang berkontribusi terhadap penyakit kardiovaskular (CVD")
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
