import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize text
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join filtered words back into a single string
    preprocessed_text = ' '.join(filtered_words)
    
    return preprocessed_text

def read_pdf(file):
    # Open the PDF file object
    reader = PdfReader(file)
    
    # Extract text from each page
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    return text

def calculate_cosine_similarity(text1, text2):
    # Preprocess text
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Create CountVectorizer object
    vectorizer = CountVectorizer().fit([preprocessed_text1, preprocessed_text2])

    # Transform text into sparse matrix
    text_matrix = vectorizer.transform([preprocessed_text1, preprocessed_text2])

    # Calculate cosine similarity
    similarity = cosine_similarity(text_matrix)

    return similarity[0, 1]
