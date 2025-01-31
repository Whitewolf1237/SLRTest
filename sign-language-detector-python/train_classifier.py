import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime  # Import datetime here

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Ensure all data points have the same length, pad with zeros if necessary
max_length = max(len(sample) for sample in data)  # Find the longest data sample

# Pad each data sample to have the same length
padded_data = []

for sample in data:
    # If the sample is shorter than the max length, pad it with zeros
    if len(sample) < max_length:
        padded_sample = sample + [0] * (max_length - len(sample))
    else:
        padded_sample = sample
    padded_data.append(padded_sample)

# Convert to numpy array
data = np.asarray(padded_data)

# Encode string labels to numeric labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Print more evaluation metrics (precision, recall, F1-score)
# Ensure target_names match the number of classes in the predictions
print("\nClassification Report:")
try:
    # Match target_names to the classes in y_test and y_predict
    print(classification_report(y_test, y_predict, target_names=label_encoder.classes_))
except ValueError as e:
    # Handle mismatch between classes in test and prediction labels
    print(f"Error in classification report: {e}")
    # You can also print unique labels here for debugging
    print(f"Unique classes in y_test: {np.unique(y_test)}")
    print(f"Unique classes in y_predict: {np.unique(y_predict)}")

# Save the trained model and label encoder
model_filename = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.p'
with open(model_filename, 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

print(f"Model and label encoder saved to {model_filename}")
