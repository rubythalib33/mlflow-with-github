import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log the experiment using MLflow
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "Logistic Regression")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the trained model
    mlflow.sklearn.log_model(model, "model")
