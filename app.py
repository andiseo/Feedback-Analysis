import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import mysql.connector
from mysql.connector import Error
import nltk  # Pastikan nltk diimpor di sini

# Mengunduh resource NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi untuk memuat model dari file pickle
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('indonesian'))
    
    # Lowercasing teks
    text = text.lower()
    
    # Tokenisasi teks
    tokens = word_tokenize(text)
    
    # Pembersihan teks dari tanda baca dan stopwords
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming teks
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Menggabungkan kembali token-token menjadi kalimat
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Fungsi untuk melakukan analisis sentimen
def analyze_sentiment(text, model, vectorizer):
    # Preprocessing teks
    preprocessed_text = preprocess_text(text)
    
    # Transformasi teks
    processed_text = vectorizer.transform([preprocessed_text])
    
    # Prediksi sentimen
    sentiment = model.predict(processed_text)[0]
    return sentiment

# Fungsi untuk menyimpan hasil analisis ke database
def save_to_database(text, sentiment):
    try:
        mysqldb = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='feedback'
        )
        if mysqldb.is_connected():
            cursor = mysqldb.cursor()
            query = "INSERT INTO sentiment_analysis (text, sentiment) VALUES (%s, %s)"
            values = (text, sentiment)
            cursor.execute(query, values)
            mysqldb.commit()
            cursor.close()
            mysqldb.close()
            return True
    except mysql.connector.Error as e:
        st.error(f"Error while connecting to MySQL: {e.msg}")
        return False
    except Exception as ex:
        st.error(f"An error occurred: {ex}")
        return False


# Load model dan vectorizer
model = load_model('logistic_regression.pkl')
vectorizer = load_model('vectorizer.pkl')

# Antarmuka Streamlit
col1, col2 = st.columns(2)

with col1:
    st.image("Hero-Image.png")

with col2:
    st.markdown("""
        <div style='margin-top: 100px;'>
            <h1 style='font-size: 30px; color: #E03168'>Terima kasih telah mengikuti pembelajaran bersama Pandusaha</h1>
        </div>
    """, unsafe_allow_html=True)

text = st.text_area('Masukkan anda sangat berharga bagi Pandusaha untuk terus mengembangkan layanan edukasi : ')
if st.button('Kirim Feedback'):
    if text:
        sentiment = analyze_sentiment(text, model, vectorizer)
        
        # Simpan hasil analisis ke database
        if save_to_database(text, sentiment):
            st.success('Terima kasih atas feedback yang Anda berikan. Feedback Anda sangat berharga untuk perbaikan layanan kami.')
        else:
            st.error('Feedback gagal terkirim, mohon coba lagi dalam beberapa saat')
    else:
        st.warning('Silakan masukkan feedback Anda sebelum mengirim.')
