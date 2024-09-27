# feature_extraction.py

from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import DataPreprocessor

class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.tfidf_vect = TfidfVectorizer(max_features=5000)

    def extract_features(self):
        X = self.tfidf_vect.fit_transform(self.df['Message']).toarray()
        y = self.df['Category'].values
        return X, y
