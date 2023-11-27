import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Predikat Kelulusan Mahasiswa Menggunakan Model Random Forest Classifier</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Ananda Ramadana Ahmad Mulya | 210411100135 | PSD - B</h4>", unsafe_allow_html=True
)

# memanggil dataset
df = pd.read_csv('dataset_baru.csv')
st.write(df)

# memisahkan kolom fitur dan kolom target
fitur = df.drop(columns=['Target'], axis =1)
target = df['Target']

# melakukan split data training dan data testing
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# memanggil file normalisasi data menggunakan minmax
with open('minmaxscaler_baru.pkl', 'rb') as file_normalisasi:
    minmax = pickle.load(file_normalisasi)
    
minmaxtraining = minmax.transform(fitur_train)
minmaxtesting = minmax.transform(fitur_test)

# memanggil file model terbaik menggunakan random forest
with open('best_model.pkl', 'rb') as model_file:
    model_rf = pickle.load(model_file)

model_rf.fit(minmaxtraining, target_train)
prediksi_target = model_rf.predict(minmaxtesting)

# prediksi
# st.warning("1 - 1st phase - general contingent 2 - Ordinance No. 612/93 3 - 1st phase - special contingent (Azores Island) 4 - Holders of other higher courses 5 - Ordinance No. 854-B/99 6 - International student (bachelor) 7 - 1st phase - special contingent (Madeira Island) 8 - 2nd phase - general contingent 9 - 3rd phase - general contingent 10 - Ordinance No. 533-A/99, item b2 (Different Plan) 11 - Ordinance No. 533-A/99, item b3 (Other Institution) 12 - Over 23 years old 13 - Transfer 14 - Change of course 15 - Technological specialization diploma holders 16 - Change of institution/course 17 - Short cycle diploma holders 18 - Change of institution/course (International)")
app_mode = st.radio("Apakah jenis metode yang anda gunakan untuk tugas terakhir?", ["none", "0", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"])

course = st.radio("Apakah jenis kursus yang pernah anda ambil?", ["none", "1","2",
                                                                  "3","4",
                                                                  "5","6",
                                                                  "7","8",
                                                                  "9","10",
                                                                  "11","12",
                                                                  "13","14",
                                                                  "15","16","17"])

prev_qual = st.radio("Apakah pendidikan terakhir anda?", ["none", "0", "1","2",
                                                          "3","4"
                                                          "5","6"
                                                          "7","8",
                                                          "9","10",
                                                          "11","12",
                                                          "13","14 ",
                                                          "15","16",
                                                          "17","18",
                                                          "19"])

father_qual = st.radio("Apakah pendidikan terakhir ayah?", ["none", "0", "1","2","3","4","5","6","7","8","9","10",
                                                            "11","12",
                                                            "13","14",
                                                            "15","16,"
                                                            "17","18",
                                                            "19","20",
                                                            "21","22",
                                                            "23","24",
                                                            "25","26",
                                                            "27","28",
                                                            "29","30",
                                                            "31","32",
                                                            "33","34"])

mother_ocup = st.radio("Apakah pekerjaan ibu?", ["none", "0 - Murid", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29"])

st.warning("Tekan 0 untuk 'tidak' dan 1 untuk 'ya'")
debtor = st.radio("Apakah anda sedang memiliki tanggungan biaya?", ["none", "0", "1"])

st.warning("Tekan 0 untuk 'tidak' dan 1 untuk 'ya'")
Tuition = st.radio("Apakah anda telah membayar biaya kuliah terakhir?", ["none", "0", "1"])

scholarship = st.number_input ('Input sertifikat yang dimiliki.')

age = st.number_input ('Input umur mahasiswa saat ini.')

Curricular_units_1sem_enr = st.number_input ('Input mata kuliah yang akan dipilih di semester pertama.')

Curricular_units_1sem_app = st.number_input ('Input mata kuliah yang telah disetujui di semester pertama.')

Curricular_units_1sem_grade = st.number_input ('Input rata - rata nilai di semester pertama.')

Curricular_units_1sem_evaluations = st.number_input ('Input mata kuliah yang telah dinilai di semester pertama.')

Curricular_units_2sem_enr = st.number_input ('Input mata kuliah yang akan dipilih di semester kedua.')

Curricular_units_2sem_app = st.number_input ('Input mata kuliah yang telah disetujui di semester kedua.')

Curricular_units_2sem_grade = st.number_input ('Input rata - rata nilai di semester kedua.')

Curricular_units_2sem_evaluations = st.number_input ('Input mata kuliah yang telah dinilai di semester kedua.')

if st.button('Cek Hasil'):
    if app_mode is not "none" and course is not "none" and prev_qual is not "none" and father_qual is not "none" and mother_ocup is not "none" and debtor is not "none" and Tuition is not "none" and scholarship is not 0.0 and age is not 0.0 and Curricular_units_1sem_enr is not 0.0 and Curricular_units_1sem_app is not 0.0 and Curricular_units_1sem_evaluations is not 0.0 and Curricular_units_1sem_grade is not 0.0 and Curricular_units_2sem_enr is not 0.0 and Curricular_units_2sem_app is not 0.0 and Curricular_units_2sem_grade is not 0.0 and Curricular_units_2sem_evaluations :
        st.text('Prediksi : ')
        prediksi = model_rf.predict([[app_mode, course, prev_qual, father_qual, mother_ocup, debtor, Tuition, scholarship, age, Curricular_units_1sem_enr, Curricular_units_1sem_app, Curricular_units_1sem_grade, Curricular_units_1sem_evaluations, Curricular_units_2sem_enr, Curricular_units_2sem_app, Curricular_units_2sem_grade, Curricular_units_2sem_evaluations]])
        if prediksi[0] == 2:
            st.success("Anda diprediksi lulus !")
        elif prediksi[0] == 1:
            st.warning("Anda diprediksi masih melanjutkan pendidikan !")
        elif prediksi[0] == 0:
            st.warning("Anda diprediksi dikeluarkan !")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')