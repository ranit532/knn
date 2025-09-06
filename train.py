import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import load_iris
import joblib

# --- Setup ---
# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

DATA_PATH = "data/dataset.csv"
ELBOW_PLOT_PATH = "images/elbow_plot.png"
CONFUSION_MATRIX_PATH = "images/confusion_matrix.png"
MODEL_PATH = "models/knn_model.joblib"

# --- 1. Dataset Creation ---
def create_dataset():
    """Loads the Iris dataset and saves it as a CSV if it doesn't exist."""
    if not os.path.exists(DATA_PATH):
        print("Creating dataset...")
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                          columns=iris['feature_names'] + ['target'])
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved to {DATA_PATH}")
    else:
        print("Dataset already exists.")

# --- Main Training Logic ---
if __name__ == "__main__":
    create_dataset()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow experiment setup
    mlflow.set_experiment("KNN_Iris_Classification")
    
    with mlflow.start_run() as run:
        print(f"Starting MLflow Run: {run.info.run_id}")

        # --- 2. Elbow Method for Optimal K ---
        print("Finding optimal k using the Elbow Method...")
        error_rate = []
        k_range = range(1, 15)
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            pred_k = knn.predict(X_test)
            error_rate.append(np.mean(pred_k != y_test))
            
        # Plotting the Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.savefig(ELBOW_PLOT_PATH)
        plt.close()
        print(f"Elbow method plot saved to {ELBOW_PLOT_PATH}")
        mlflow.log_artifact(ELBOW_PLOT_PATH)

        # --- 3. Train Final Model ---
        # Choosing k=5 based on the elbow plot (a common choice for this dataset)
        optimal_k = 5
        print(f"Training final model with k={optimal_k}...")
        knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
        knn_final.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("n_neighbors", optimal_k)
        
        # --- 4. Evaluate Model ---
        print("Evaluating model...")
        y_pred = knn_final.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

        # --- 5. Generate and Log Confusion Matrix ---
        print("Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=load_iris().target_names, yticklabels=load_iris().target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close()
        print(f"Confusion matrix plot saved to {CONFUSION_MATRIX_PATH}")
        mlflow.log_artifact(CONFUSION_MATRIX_PATH)
        
        # --- 6. Save and Log Model ---
        print(f"Saving model to {MODEL_PATH}")
        joblib.dump(knn_final, MODEL_PATH)
        
        # Log the model in MLflow
        mlflow.sklearn.log_model(knn_final, "knn-model")
        
        print("\nMLflow run completed successfully!")
        print(f"To view the experiment, run: mlflow ui")