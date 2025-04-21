import os
import numpy as np
import cv2
import argparse
import pickle
from sklearn.model_selection import train_test_split
from models.expression_model import ExpressionRecognitionModel
from models.gesture_model import GestureRecognitionModel

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train Custom Models for Expression and Gesture Recognition')

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the training data')

    parser.add_argument('--model_type', type=str, required=True, choices=['expression', 'gesture'],
                        help='Type of model to train (expression or gesture)')

    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained model')

    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')

    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')

    return parser.parse_args()

def load_expression_data(data_dir):
    """
    Load facial expression data from the specified directory.

    Args:
        data_dir: Directory containing the expression data.

    Returns:
        X: List of face images.
        y: List of expression labels.
        class_names: List of class names.
    """
    expression_dir = os.path.join(data_dir, 'expression')
    if not os.path.exists(expression_dir):
        raise ValueError(f"Expression data directory {expression_dir} does not exist")

    X = []
    y = []
    class_names = []

    # Iterate through expression directories
    for class_idx, class_name in enumerate(sorted(os.listdir(expression_dir))):
        class_dir = os.path.join(expression_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_names.append(class_name)
        print(f"Loading {class_name} expressions...")

        # Iterate through images in the class directory
        for filename in os.listdir(class_dir):
            if not filename.endswith('.jpg'):
                continue

            # Load image
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Resize to a standard size
                img = cv2.resize(img, (48, 48))
                X.append(img)
                y.append(class_idx)

    return X, y, class_names

def load_gesture_data(data_dir):
    """
    Load hand gesture data from the specified directory.

    Args:
        data_dir: Directory containing the gesture data.

    Returns:
        X: List of hand landmark features.
        y: List of gesture labels.
        class_names: List of class names.
    """
    gesture_dir = os.path.join(data_dir, 'gesture')
    if not os.path.exists(gesture_dir):
        raise ValueError(f"Gesture data directory {gesture_dir} does not exist")

    X = []
    y = []
    class_names = []

    # Iterate through gesture directories
    for class_idx, class_name in enumerate(sorted(os.listdir(gesture_dir))):
        class_dir = os.path.join(gesture_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_names.append(class_name)
        print(f"Loading {class_name} gestures...")

        # Iterate through landmark files in the class directory
        for filename in os.listdir(class_dir):
            if not filename.endswith('.npy'):
                continue

            # Load landmarks
            landmarks_path = os.path.join(class_dir, filename)
            landmarks = np.load(landmarks_path)

            X.append(landmarks)
            y.append(class_idx)

    return X, y, class_names

def train_expression_model(args):
    """
    Train a facial expression recognition model.

    Args:
        args: Command line arguments.
    """
    # Load data
    X, y, class_names = load_expression_data(args.data_dir)

    if len(X) == 0:
        print("No expression data found. Please collect data first.")
        return

    print(f"Loaded {len(X)} expression samples with {len(class_names)} classes")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Initialize model
    model = ExpressionRecognitionModel()

    # Update class labels
    model.class_labels = class_names
    model.num_classes = len(class_names)

    # Train the model
    print("Training expression model...")
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(args.output_dir, 'custom_expression_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model.model, 'scaler': model.scaler, 'class_labels': model.class_labels}, f)

    print(f"Expression model trained and saved to {model_path}")

def train_gesture_model(args):
    """
    Train a hand gesture recognition model.

    Args:
        args: Command line arguments.
    """
    # Load data
    X, y, class_names = load_gesture_data(args.data_dir)

    if len(X) == 0:
        print("No gesture data found. Please collect data first.")
        return

    print(f"Loaded {len(X)} gesture samples with {len(class_names)} classes")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Initialize model
    model = GestureRecognitionModel()

    # Update class labels
    model.class_labels = class_names
    model.num_classes = len(class_names)

    # Train the model
    print("Training gesture model...")
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(args.output_dir, 'custom_gesture_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model.model, 'scaler': model.scaler, 'class_labels': model.class_labels}, f)

    print(f"Gesture model trained and saved to {model_path}")

def main():
    """
    Main function for training custom models.
    """
    # Parse arguments
    args = parse_args()

    # Train the specified model
    if args.model_type == 'expression':
        train_expression_model(args)
    elif args.model_type == 'gesture':
        train_gesture_model(args)

if __name__ == '__main__':
    main()
