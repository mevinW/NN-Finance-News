import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# 1. Load and preprocess the data

# Assume training_data is a DataFrame with 'text' for article and 'label' for sentiment
# ('negative', 'neutral', 'positive')
def preprocess_data(training_data, max_words=5000, max_len=200):
    # Convert sentiment labels to numerical values (0: negative, 1: neutral, 2: positive)
    label_encoder = LabelEncoder()
    training_data['label'] = label_encoder.fit_transform(training_data['label'])

    # One-hot encoding the labels
    labels = to_categorical(training_data['label'], num_classes=3)

    # Tokenize the text data
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(training_data['text'])
    word_index = tokenizer.word_index

    # Convert the text to sequences
    sequences = tokenizer.texts_to_sequences(training_data['text'])

    # Pad sequences to ensure uniform input size
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    return padded_sequences, labels, tokenizer


# 2. Build the Neural Network
def build_model(max_words=5000, max_len=200):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    #model.add(LSTM(100, dropout=0, recurrent_dropout=0))
    model.add(Dense(3, activation='softmax'))  # Multi-class classification (positive, neutral, negative)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


# 3. Train the Model
def train_model(model, padded_sequences, labels, batch_size=64, epochs=100, validation_split=0.2):
    model.fit(padded_sequences, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
              verbose=2)


# 4. Evaluate and Predict on new data
def evaluate_model(model, padded_sequences, labels):
    loss, accuracy = model.evaluate(padded_sequences, labels, verbose=2)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def predict_sentiment(model, new_articles, tokenizer, max_len=200):
    # Convert the new article text into sequences and pad them
    new_sequences = tokenizer.texts_to_sequences(new_articles)
    new_padded = pad_sequences(new_sequences, maxlen=max_len, padding='post')

    # Predict the sentiment
    predictions = model.predict(new_padded)
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_classes = np.argmax(predictions, axis=1)

    return [sentiment_labels[p] for p in predicted_classes]


# Main function to execute the whole process
def main(training_data):
    # Preprocess data
    max_words = 1000  # Number of unique words to consider
    max_len = 20  # Maximum length of each sequence (article)
    padded_sequences, labels, tokenizer = preprocess_data(training_data, max_words, max_len)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Build the neural network model
    model = build_model(max_words, max_len)

    # Train the model
    train_model(model, X_train, y_train)

    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test)

    # Example: Predict sentiment of new articles
    new_articles = ["Stock market today: Dow tumbles 400 points, tech leads Nasdaq, S&P 500 lower as 10-year yield tops 4%. US stocks slipped on Monday and the 10-year Treasury yield (^TNX) jumped past 4% for the first time since August ahead of a week of key inflation data and the start of earnings season.",
                    "Goldman Sachs lifts S&P 500 index target for year-end, next 12 months. ",
                    "SPY seems to be stable."]
    predictions = predict_sentiment(model, new_articles, tokenizer)
    print(predictions)


# Example usage
if __name__ == "__main__":
    # Assuming training_data is loaded as a DataFrame with columns 'text' and 'label'
    # For example:
    # training_data = pd.DataFrame({'text': ['SPY is expected to rise', 'SPY is going down', 'SPY is stable'],
    #                               'label': ['positive', 'negative', 'neutral']})

    # Load your own data here
    training_data = pd.read_csv('test4.csv')  # Replace with actual path to CSV file

    main(training_data)
