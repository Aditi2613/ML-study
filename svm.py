# Import necessary libraries
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Weight": [120, 150, 180, 140, 160, 200],
    "Color": ["Red", "Red", "Red", "Yellow", "Yellow", "Yellow"],
    "Class": ["Apple", "Apple", "Apple", "Banana", "Banana", "Banana"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Map "Color" to numerical values (e.g., Red: 0, Yellow: 1)
df["Color"] = df["Color"].map({"Red": 0, "Yellow": 1})

# Prepare the input features and target variable
X = df[["Weight", "Color"]]
y = df["Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = svm.SVC(kernel="linear")  # We use a linear kernel for simplicity

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict new samples
new_samples = [[170, 1], [130, 0]]  # New fruit samples to predict
predictions = clf.predict(new_samples)
print("Predictions for new samples:", predictions)
