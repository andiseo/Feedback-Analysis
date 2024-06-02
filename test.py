import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

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

# Fungsi untuk menyimpan hasil analisis ke database SQLite
def save_to_database(text, sentiment):
    try:
        # Membuat atau terhubung ke database SQLite
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        
        # Membuat tabel jika belum ada
        cursor.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT)''')

        # Menyimpan data ke database
        cursor.execute("INSERT INTO sentiment_analysis (text, sentiment) VALUES (?, ?)", (text, sentiment))
        
        # Commit perubahan
        conn.commit()
        
        # Menutup koneksi
        cursor.close()
        conn.close()
        
        return True
    except Exception as ex:
        st.error(f"An error occurred: {ex}")
        return False

# Fungsi untuk menghapus data dari database
def delete_data_from_database():
    try:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sentiment_analysis")
        conn.commit()
        conn.close()
        st.success("All data has been deleted.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Load model dan vectorizer
model = load_model('logistic_regression.pkl')
vectorizer = load_model('vectorizer.pkl')

# Define the different pages as functions
def feedback_page():
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

def display_page():
    # Fungsi untuk mengambil data dari database
    def get_data_from_database():
        try:
            conn = sqlite3.connect('feedback.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sentiment_analysis")
            data = cursor.fetchall()
            conn.close()
            return data
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    # Ambil data dari database
    data = get_data_from_database()

    # Tampilkan data jika ada
    if data:
        # Konversi data ke DataFrame
        df = pd.DataFrame(data, columns=['ID', 'Feedback', 'Sentiment'])

        # Menampilkan tabel interaktif dengan st.table()
        st.write('## Data Sentimen:')
        st.table(df)

        # Visualisasi jumlah sentimen dengan diagram lingkaran yang lebih menarik
        sentiment_counts = df['Sentiment'].value_counts()
        st.write('## Visualisasi Jumlah Sentimen:')
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = sns.color_palette('pastel')[0:len(sentiment_counts)]
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        ax.set_title('Persentase Sentimen', fontsize=16)
        st.pyplot(fig)


        # Tambahkan tombol untuk menghapus data
        if st.button('Delete All Data'):
            delete_data_from_database()
            # Refresh the data after deletion
            data = get_data_from_database()
            if data:
                df = pd.DataFrame(data, columns=['ID', 'Feedback', 'Sentiment'])
                st.table(df)
            else:
                st.write("No data available.")
    else:
        st.write('Tidak ada data yang tersedia.')

# Create a list of page names and corresponding functions
pages = {
    "Feedback": feedback_page,
    "Display": display_page
}

# Use a selectbox for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# Call the function for the selected page
pages[selection]()
