# Import necessary libraries
import pandas as pd
import numpy as np
from PEDA import PEDA  
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv("C:/Data Science and AI/Project/House Price Prediction/Datasets/Housing.csv")

# Database setup
user = 'project'
pw = 7016
db = 'HPP_Database'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Export the dataset to SQL
dataset.to_sql('housing_dataset', con=engine, if_exists='replace', chunksize=1000, index=False)

# Query the database to load the data into pandas DataFrame
sql = 'SELECT * FROM housing_dataset'
df = pd.read_sql_query(sql, con=engine)

# Boxplot for detecting outliers
sns.boxplot(df)
plt.show()

# Get numerical columns
numerical_columns = df.select_dtypes(include='number').columns.tolist()

# Initialize the Winsorizer for numerical columns
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=numerical_columns
)

# Apply winsorization to cap the outliers
df = winsor.fit_transform(df)

# Build in function for EDA
v = PEDA( df )

# Save the Isights
v.to_excel('Cleaned Business & Statical Insights.xlsx', index=False)

# Label Encoding for categorical columns
encoder = LabelEncoder()
object_columns = df.select_dtypes(include=['object']).columns

# Apply Label Encoding for each categorical column
for column in object_columns:
    df[column] = encoder.fit_transform(df[column])

# Define features (X) and target (Y)
X = df.drop(columns=['price'])
Y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# Build a Neural Network Model using TensorFlow/Keras
model = Sequential()

# Input Layer (we set the input shape to be the number of features in the dataset)
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))

# Output Layer (single output node for predicting price)
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, Y_train, epochs=100, batch_size=52, validation_data=(X_test_scaled, Y_test), verbose=2)

# Save the model to a file (e.g., 'house_price_model.h5')
model.save("house_price_model.h5")
print("Model saved as 'house_price_model.h5'")

# Evaluate the model on the test data
test_loss = model.evaluate(X_test_scaled, Y_test, verbose=0)
print(f'Test Loss (Mean Squared Error): {test_loss}')

# Make predictions on the test data
predictions = model.predict(X_test_scaled)

# Calculate R-squared
r2 = r2_score(Y_test, predictions)
print(f"R-Squared: {r2}")


# Visualizing Actual vs Predicted and saving the plot
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predictions, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.savefig("actual_vs_predicted_plot.png")  # Save the plot
plt.show()
print("Actual vs Predicted plot saved as 'actual_vs_predicted_plot.png'")

# Plot the training and validation loss over epochs and save the plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.savefig("training_validation_loss_plot.png")  # Save the plot
plt.show()
print("Training and Validation Loss plot saved as 'training_validation_loss_plot.png'")
