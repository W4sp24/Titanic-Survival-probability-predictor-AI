import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import urllib.request
from urllib.error import URLError, HTTPError
import socket
import time


# Define a function to retry URL requests
def url_get(url, retries=3, backoff_factor=2):
    """
    Fetches a URL with retry logic.

    Args:
        url (str): The URL to fetch.
        retries (int): Number of times to retry.
        backoff_factor (int): The factor by which to increase the delay between retries.

    Returns:
        The response object if successful, None otherwise.
    """
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(url)
        except (URLError, HTTPError, socket.gaierror) as e:
            if attempt < retries - 1:  # Only wait if there are more retries
                wait_time = backoff_factor * (2 ** attempt)
                print(f"Error accessing {url}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to retrieve {url} after {retries} attempts.  Error: {e}")
                raise  # Re-raise the exception after all retries fail
    return None  #Should never reach here.

# Load the datasets with retry
try:
    dftrain = pd.read_csv(url_get('https://storage.googleapis.com/tf-datasets/titanic/train.csv'))
    dfeval = pd.read_csv(url_get('https://storage.googleapis.com/tf-datasets/titanic/eval.csv'))
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()  # Exit if the dataset cannot be loaded

# Separate features and labels
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# 1. Preprocess the Data

# Define categorical and numerical features
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Preprocess categorical features
def preprocess_categorical(df, columns):
    processed_dfs = []
    for col in columns:
        # Convert the column to string type first to handle potential numerical errors and NaNs
        df[col] = df[col].astype(str)
        dummies = pd.get_dummies(df[col], prefix=col)
        processed_dfs.append(dummies)
    return pd.concat(processed_dfs, axis=1)

# Preprocess numerical features
def preprocess_numerical(df, columns):
    processed_dfs = []
    for col in columns:
        #  Handle missing values by filling them with the mean
        df[col] = df[col].fillna(df[col].mean())
        processed_dfs.append(df[col])
    return pd.concat(processed_dfs, axis=1)

# Apply preprocessing
categorical_processed_train = preprocess_categorical(dftrain, CATEGORICAL_COLUMNS)
categorical_processed_eval = preprocess_categorical(dfeval, CATEGORICAL_COLUMNS)
numerical_processed_train = preprocess_numerical(dftrain, NUMERIC_COLUMNS)
numerical_processed_eval = preprocess_numerical(dfeval, NUMERIC_COLUMNS)

# Combine the processed features.
X_train = pd.concat([categorical_processed_train, numerical_processed_train], axis=1)
X_eval = pd.concat([categorical_processed_eval, numerical_processed_eval], axis=1)

# Align the columns
train_cols = set(X_train.columns)
eval_cols = set(X_eval.columns)

missing_in_train = eval_cols - train_cols
missing_in_eval = train_cols - eval_cols

for col in missing_in_train:
    X_train[col] = 0  # Or use another appropriate value
for col in missing_in_eval:
    X_eval[col] = 0  # Or use another appropriate value

# Ensure both DataFrames have the same columns in the same order
X_eval = X_eval[list(X_train.columns)]


# 2. Create the Model
# Build the model using the Functional API
inputs = []
for name in X_train.columns:
    input_tensor = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)
    inputs.append(input_tensor)



concatenated_inputs = layers.Concatenate()(inputs)
dense1 = layers.Dense(64, activation='relu')(concatenated_inputs)
dense2 = layers.Dense(32, activation='relu')(dense1)
output = layers.Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=inputs, outputs=output)


# 3. Train the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Convert data to dataset format for the model
def df_to_dataset(df, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(df), y))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

batch_size = 32
train_ds = df_to_dataset(X_train, y_train, batch_size)
eval_ds = df_to_dataset(X_eval, y_eval, batch_size)

# Train the model
model.fit(train_ds, epochs=20)

# 4. Evaluate the Model
loss, accuracy = model.evaluate(eval_ds)
print(f"Accuracy: {accuracy:.4f}")



# 5. Make Predictions
# Example: Predict survival for new passengers
new_passengers = pd.DataFrame({
    'sex': ['male', 'female', 'male'],
    'n_siblings_spouses': [1, 0, 2],
    'parch': [0, 1, 0],
    'class': ['Third', 'First', 'Second'],
    'deck': ['unknown', 'C', 'unknown'],
    'embark_town': ['Southampton', 'Cherbourg', 'Queenstown'],
    'alone': ['n', 'n', 'n'],
    'age': [25, 35, 45],
    'fare': [15.00, 100.00, 25.00]
})

# Preprocess the new data in the same way as the training data
categorical_processed_new = preprocess_categorical(new_passengers, CATEGORICAL_COLUMNS)
numerical_processed_new = preprocess_numerical(new_passengers, NUMERIC_COLUMNS)
X_new = pd.concat([categorical_processed_new, numerical_processed_new], axis=1)

# Align the columns
train_cols = list(X_train.columns)
new_cols = list(X_new.columns)

missing_in_new = list(set(train_cols) - set(new_cols))
for col in missing_in_new:
    X_new[col] = 0
X_new = X_new[train_cols]  # Ensure the columns are in the same order as training data

# Convert to dataset
new_ds = tf.data.Dataset.from_tensor_slices(dict(X_new))
new_ds = new_ds.batch(32)  # Batch size doesn't matter for prediction usually

# Make predictions
predictions = model.predict(new_ds)
# Convert probabilities to binary predictions (0 or 1)
binary_predictions = (predictions > 0.5).astype(int)

print("\nPredictions for new passengers:")
for i, pred in enumerate(binary_predictions):
    print(f"Passenger {i+1}: Survived - {pred[0]}")
