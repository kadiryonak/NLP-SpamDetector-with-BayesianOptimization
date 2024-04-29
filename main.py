# main.py
from data_preprocessing import DataPreprocessor

def main():
    
    file_path = r"File Path"    
    preprocessor = DataPreprocessor(file_path)    
    df_processed = preprocessor.load_and_preprocess_data()    
    print(df_processed.head())
    
if __name__ == "__main__":
    main()
