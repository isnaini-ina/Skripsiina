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

st.set_page_config(page_title="SKRIPSI-ina", page_icon='utm.png')
@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)



with st.sidebar:
    selected = option_menu('',['Home', 'Research', 'Dataset', 'Preprocessing', 'Modelling', 'Implementation'], default_index=0)
if (selected == 'Home'):
    st.markdown("<h1 style='text-align: center; '>Implementasi Metode Smote Pada Klasifikasi Penyakit Stroke dengan Algoritma Naive Bayes</h1>", unsafe_allow_html=True)
    url_logo = 'utm.png'
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
            "Data yang digunakan diperoleh dari repositori Kaggle yang terdiri dari catatan kesehatan yang dikumpulkan dari berbagai rumah sakit di Bangladesh oleh tim peneliti untuk tujuan akademis. Data ini dapat diakses secara publik melalui https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
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
