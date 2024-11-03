import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

# 1. Load and preprocess the data

# Assume training_data is a DataFrame with 'text' for article and 'label' for sentiment (1 = positive, 0 = negative)
def preprocess_data(training_data, max_words=5000, max_len=200):
    # Convert sentiment labels to numerical values
    le = LabelEncoder()
    training_data['label'] = le.fit_transform(training_data['label'])

    # Tokenize the text data
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(training_data['text'])
    word_index = tokenizer.word_index

    # Convert the text to sequences
    sequences = tokenizer.texts_to_sequences(training_data['text'])

    # Pad sequences to ensure uniform input size
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    return padded_sequences, training_data['label'], tokenizer

# 2. Build the Neural Network
def build_model(max_words=5000, max_len=200):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid')) # Binary classification (positive/negative)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# 3. Train the Model
def train_model(model, padded_sequences, labels, batch_size=64, epochs=50, validation_split=0.2):
    model.fit(padded_sequences, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=2)

# 4. Evaluate and Predict on new data
def evaluate_model(model, padded_sequences, labels):
    loss, accuracy = model.evaluate(padded_sequences, labels, verbose=2)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

def predict_sentiment(model, new_articles, tokenizer, max_len=200):
    # Convert the new article text into sequences and pad them
    new_sequences = tokenizer.texts_to_sequences(new_articles)
    new_padded = pad_sequences(new_sequences, maxlen=max_len, padding='post')

    # Predict the sentiment
    predictions = model.predict(new_padded)
    return ['Positive' if p > 0.5 else 'Negative' for p in predictions]

# Main function to execute the whole process
def main(training_data):
    # Preprocess data
    max_words = 20 # Number of unique words to consider
    max_len = 20 # Maximum length of each sequence (article)
    padded_sequences, labels, tokenizer = preprocess_data(training_data, max_words, max_len)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.1, random_state=42)

    # Build the neural network model
    model = build_model(max_words, max_len)

    # Train the model
    train_model(model, X_train, y_train)

    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test)

    # Example: Predict sentiment of new articles
    new_articles = ["SPY is expected to go up due to strong earnings.",
                    "Strong economic data suggests SPY might grow.",
                    "Weak economic data suggests SPY is going down."]
    predictions = predict_sentiment(model, new_articles, tokenizer)
    print(new_articles)
    print(predictions)

# Example usage
if __name__ == "__main__":
    # Assuming training_data is loaded as a DataFrame with columns 'text' and 'label'
    # For example:
    training_data = pd.DataFrame({'text': ['SPY is expected to rise',
                                           'SPY is going down',
                                           'SPY is expected to grow',
                                           'SPY is expected to increase',
                                           'SPY is predicted to rise',
                                           'SPY will go down',
                                           'SPY is going up',
                                           'SPY will rise',
                                           'SPY will grow',
                                           'SPY will be down',
                                           'SPY is going down'],
                                  'label': ['positive',
                                            'negative',
                                            'positive',
                                            'positive',
                                            'positive',
                                            'negative',
                                            'positive',
                                            'positive',
                                            'positive',
                                            'negative',
                                            'negative']})

    # Load your own data here
    # training_data = pd.read_csv('test.csv') # Replace with actual path to CSV file

    main(training_data)
