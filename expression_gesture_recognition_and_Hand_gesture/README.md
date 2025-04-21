# Enhanced Facial Expression and Hand Gesture Recognition System

A standalone Python application capable of recognizing facial expressions and hand gestures in real-time using a webcam, with advanced features, optimizations, and an enhanced UI. The system now includes improved hand landmark detection and finger counting capabilities.

## Features

- Real-time face detection and facial expression recognition
- Real-time hand detection and gesture recognition
- Accurate finger counting and hand landmark detection
- Advanced feature extraction for improved recognition accuracy
- Enhanced UI with multiple display modes
- Performance optimizations for better real-time processing
- Data collection tools for training custom models
- Support for a wider range of expressions and gestures

## Technology Stack

- OpenCV - Webcam feed, image processing, and face detection
- Scikit-learn - Machine learning models for recognition
- NumPy - Numerical processing
- Scikit-image - Advanced feature extraction

## Project Structure

```
expression_gesture_recognition/
│
├── data/                   # Datasets or collected frames
├── models/                 # Trained models
│   ├── expression_model.py # Facial expression recognition model
│   └── gesture_model.py    # Hand gesture recognition model
├── utils/                  # Utility functions
│   ├── face_utils.py       # Face detection utilities
│   ├── hand_utils.py       # Hand detection utilities
│   ├── improved_hand_utils.py # Improved hand landmark detection
│   ├── precise_hand_detector.py # Precise hand detection with bounding box
│   ├── visualization.py    # Basic visualization utilities
│   ├── advanced_features.py # Advanced feature extraction
│   └── optimization.py     # Performance optimization utilities
├── main.py                 # Basic application entry point
├── enhanced_ui.py          # Enhanced UI application
├── optimized_main.py       # Optimized application with all enhancements
├── finger_counter.py       # Finger counting application
├── finger_counting_app.py  # Dedicated finger counting application
├── advanced_gesture_recognition.py # Advanced gesture recognition with landmarks
├── hand_detection_app.py    # Precise hand detection application
├── test_hand_detector.py   # Test hand detector on images
├── collect_training_data.py # Tool for collecting training data
├── train_model.py          # Scripts for training models
├── train_custom_models.py  # Scripts for training custom models
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Application
```
python main.py
```

### Enhanced UI Application
```
python enhanced_ui.py
```

### Optimized Application
```
python optimized_main.py [options]
```

Options:
- `--use_advanced_features`: Use advanced feature extraction
- `--use_threading`: Use threading for frame processing
- `--use_motion_detection`: Use motion detection to optimize processing
- `--downscale_factor`: Downscale factor for frame processing (percentage)
- `--ui_mode`: UI mode (default, debug, minimal)

### Finger Counting Application
```
python finger_counting_app.py
```

Options:
- `--threshold`: Threshold for hand detection (default: 60)
- `--blur`: Blur value for hand detection (default: 41)

### Advanced Gesture Recognition
```
python advanced_gesture_recognition.py [options]
```

Options:
- `--threshold`: Threshold for hand detection
- `--blur`: Blur value for hand detection
- `--record`: Enable recording of detected gestures
- `--output_dir`: Directory to save recorded gestures

### Precise Hand Detection
```
python hand_detection_app.py [options]
```

Options:
- `--camera`: Camera device index (default: 0)
- `--width`: Camera frame width (default: 640)
- `--height`: Camera frame height (default: 480)
- `--min_detection_confidence`: Minimum confidence for detection (default: 0.7)
- `--min_tracking_confidence`: Minimum confidence for tracking (default: 0.5)

### Test Hand Detection on Images
```
python test_hand_detector.py --image path/to/your/image.jpg [options]
```

Options:
- `--output`: Path to save the output image (default: output.jpg)
- `--min_detection_confidence`: Minimum confidence for detection (default: 0.7)

### Data Collection
```
python collect_training_data.py --data_type [expression|gesture] --output_dir data --num_samples [count]
```

### Training Models
```
python train_model.py --model_type [expression|gesture] --data_path [path] --output_path [path]
```

### Training Custom Models
```
python train_custom_models.py --model_type [expression|gesture] --data_dir data --output_dir models
```

### Using Custom Models
```
python main.py --expression_model models/custom_expression_model.pkl --gesture_model models/custom_gesture_model.pkl
```

## Supported Expressions and Gestures

### Facial Expressions
- Happy
- Sad
- Angry
- Surprised
- Neutral
- Fearful
- Disgusted
- Contempt
- Confused
- Calm
- Shocked
- Wink
- Depressed

### Hand Gestures
- Thumbs up
- Thumbs down
- Peace sign
- Open palm
- Closed fist
- Pointing
- OK sign
- Wave
- Grab
- Pinch
- Swipe left
- Swipe right
- Super (shaka sign)
- I love you

## Advanced Features

### Enhanced Feature Extraction
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Gabor filters
- Haar-like features
- Contour-based features

### Hand Landmark Detection
- Accurate finger counting (0-5 fingers)
- Hand landmark identification and tracking
- Convexity defects for finger separation
- Gesture recognition based on finger positions
- Visual feedback of detected landmarks
- Precise hand detection with green bounding box and confidence display
- Real-time gesture classification (hello, peace, thumbs up, fist, pointing)

### Performance Optimizations
- Multi-threading for parallel processing
- Motion detection to focus processing on relevant regions
- Frame downscaling for faster processing
- Optimized detection parameters

### Enhanced UI
- Multiple display modes (default, debug, minimal)
- Real-time performance metrics
- Confidence visualization
- History tracking
- Dedicated finger counting display

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
