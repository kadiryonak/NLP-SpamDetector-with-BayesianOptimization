# data_preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_preprocess_data(self):
        # nltk modülleri indiriliyor
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        df = pd.read_csv(self.file_path, encoding='latin-1')
        df.drop_duplicates(inplace=True)
        df['Message'] = df['Message'].apply(lambda x: x.lower()) 
        df['Message'] = df['Message'].apply(lambda x: re.sub(r'\d+', '', x))
        df['Message'] = df['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        
        df['Message'] = df['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        df['Message'] = df['Message'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        df['Message'] = df['Message'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

        le = LabelEncoder()
        df['Category'] = le.fit_transform(df['Category'])
        
        return df

