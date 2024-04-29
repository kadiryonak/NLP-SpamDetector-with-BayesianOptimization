# main.py
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor

def main():
    
    file_path = r"File Path"    
    preprocessor = DataPreprocessor(file_path)    
    df_processed = preprocessor.load_and_preprocess_data()    
    print(df_processed.head())
    feature_extractor = FeatureExtractor(df_processed)
    X, y = feature_extractor.extract_features()

if __name__ == "__main__":
    main()
