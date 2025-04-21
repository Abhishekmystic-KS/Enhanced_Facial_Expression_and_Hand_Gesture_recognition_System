import os
import numpy as np
import pickle
from models.expression_model import ExpressionRecognitionModel
from models.gesture_model import GestureRecognitionModel

def train_expression_model():
    """
    Train and save an expression recognition model.
    """
    # Create a model
    model = ExpressionRecognitionModel()

    # Update class labels to match your trained expressions
    model.class_labels = ['Happy', 'Sad', 'Surprise', 'Shock', 'Wink', 'Depressed']
    model.num_classes = len(model.class_labels)

    # Create dummy training data
    # In a real scenario, you would use your actual training data
    num_samples = 100
    feature_dim = 2309  # Exactly match the runtime feature dimension

    # Generate random features
    X_train = np.random.random((num_samples, feature_dim))

    # Generate random labels
    y_train = np.random.randint(0, model.num_classes, num_samples)

    # Train the model
    model.train(X_train, y_train)

    # Create output directory if it doesn't exist
    os.makedirs('../trained_models', exist_ok=True)

    # Save the model
    model_path = '../trained_models/expression_model.pkl'
    model.save_model(model_path)

    print(f"Expression model trained and saved to {model_path}")

def train_gesture_model():
    """
    Train and save a gesture recognition model.
    """
    # Create a model
    model = GestureRecognitionModel()

    # Update class labels to match your trained gestures
    model.class_labels = ['Thumbs Up', 'Thumbs Down', 'Super', 'I Love You']
    model.num_classes = len(model.class_labels)

    # Create dummy training data
    # In a real scenario, you would use your actual training data
    num_samples = 100
    feature_dim = 63  # Expected feature dimension

    # Generate random features
    X_train = np.random.random((num_samples, feature_dim))

    # Generate random labels
    y_train = np.random.randint(0, model.num_classes, num_samples)

    # Train the model
    model.train(X_train, y_train)

    # Create output directory if it doesn't exist
    os.makedirs('../trained_models', exist_ok=True)

    # Save the model
    model_path = '../trained_models/gesture_model.pkl'
    model.save_model(model_path)

    print(f"Gesture model trained and saved to {model_path}")

if __name__ == "__main__":
    train_expression_model()
    train_gesture_model()
