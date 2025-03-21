import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import keras
from keras import layers

def train_neural_network_classifier(data_path: str):
    all_data = pd.read_parquet(data_path)
    # Ensure the target variable is correctly encoded
    label_encoder = preprocessing.LabelEncoder()
    all_data['rating'] = label_encoder.fit_transform(all_data['rating'])
    x = all_data[['rsi', 'trix', 'stoch', 'ppo', 'vzo']]
    y = all_data['rating']
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = keras.Sequential([
        layers.InputLayer(input_shape=(5,)),
        layers.Dense(64, activation='relu'),      # Hidden layer with 64 neurons
        layers.Dense(32, activation='relu'),      # Hidden layer with 32 neurons
        layers.Dense(5, activation='softmax')     # Output layer with 5 classes (categorical output)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Print the model summary
    model.summary()
    model.fit(x_train, y_train, epochs=1)
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(x_test[:3])
    print("predictions shape:", predictions.shape)
    return model, label_encoder
    
    
                
                
def train_xgboost_classifier(data_dir: str) -> xgb.XGBClassifier:
    all_data = pd.DataFrame()
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_processed.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
                all_data = pd.concat([all_data, df])
                
    # Ensure the target variable is correctly encoded
    label_encoder = preprocessing.LabelEncoder()
    all_data['rating'] = label_encoder.fit_transform(all_data['rating'])

    # Prepare the feature and target variables
    X = all_data[['rsi', 'trix', 'stoch', 'ppo', 'vzo']]
    y = all_data['rating']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost classifier
    model = xgb.XGBClassifier(learning_rate=0.001, iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'XGBoost Model Accuracy: {accuracy}')

    return model, label_encoder

def plot_signals(data: pd.DataFrame, title: str):
    # Plotting the open price
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['open'], label='Open Price', color='blue')

    # Plotting the signals
    buy_signals = data[data['rating'] == 'buy']
    mega_buy_signals = data[data['rating'] == 'mega buy']
    sell_signals = data[data['rating'] == 'sell']
    mega_sell_signals = data[data['rating'] == 'mega sell']

    plt.scatter(buy_signals.index, buy_signals['open'], label='Buy', color='yellowgreen', marker='^', alpha=1)
    plt.scatter(mega_buy_signals.index, mega_buy_signals['open'], label='Mega Buy', color='green', marker='^', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['open'], label='Sell', color='orange', marker='v', alpha=1)
    plt.scatter(mega_sell_signals.index, mega_sell_signals['open'], label='Mega Sell', color='red', marker='v', alpha=1)

    plt.title('Stock Open Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'./charts/test_signals_{title}.png', dpi=300, format='png')
    plt.show()
    
def plot_test_data(file: str = 'HistoricalData_1738966591583_processed.csv', output_dir: str = './data/output/'):
    test_data = os.path.join(output_dir, file)
    if os.path.exists(test_data):
        sample_data = pd.read_csv(test_data, index_col="Date", parse_dates=True)
        plot_signals(sample_data, file)
    else:
        print(f"Test data not found: {test_data}")
        
def save_models(model, name):
    # Save the models
    if not os.path.exists('./models'):
        os.makedirs('./models')
    joblib.dump(model, f'./models/{name}.pkl')

    print('Models saved to ./models directory')
    
def save_label_encoder(label_encoder, name):
    # Save the models
    if not os.path.exists('./models'):
        os.makedirs('./models')
    joblib.dump(label_encoder, f'./models/{name}.pkl')

    print('Label encoder saved to ./models directory')