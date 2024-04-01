
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor

# Separate raw text data and tags
messages = df['Message']
labels = df['Category']

# Separating training and test sets
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Tokenizer creation and fitting on training data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text to a numeric array
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Make arrays fixed length with padding
max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

from tensorflow.keras.callbacks import EarlyStopping

#LSTM Model
def lstm(dropout_rate, recurrent_dropout_rate, units):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_len))
    model.add(LSTM(units=int(units), dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Model training
    history = model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    #Early stopping overfitting i önlemek için
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    
    # Returning the truth value
    accuracy = np.amax(history.history['val_accuracy'])
    return accuracy 

# Optimization parameters
pbounds = {'dropout_rate': (0.0, 0.5), 'recurrent_dropout_rate': (0.0, 0.5), 'units': (50, 150)}

# Initialize the Bayesian Optimization object
optimizer = BayesianOptimization(
    f=lstm_opt,
    pbounds=pbounds,
    random_state=1,
)

# Start optimization
optimizer.maximize(init_points=2, n_iter=3)

# Printing the best parameters
print(optimizer.max)

# Rebuild and train the model with the best parameters
best_params = optimizer.max['params']
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_len))
model.add(LSTM(units=int(best_params['units']), dropout=best_params['dropout_rate'], recurrent_dropout=best_params['recurrent_dropout_rate']))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Retraining the model with the best parameters
model.fit(X_train_pad, y_train, epochs=4, batch_size=32, validation_split=0.2,callbacks = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True))

