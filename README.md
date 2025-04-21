# Enhanced Facial Expression and Hand Gesture Recognition System

A real-time Python application for recognizing facial expressions and hand gestures using a webcam. The system features advanced hand landmark detection, finger counting capabilities, and an enhanced UI.

## How it works?
https://github.com/user-attachments/assets/7e8b4e7e-483a-4980-830c-d86b58c324cf


## 🌟 Features

- **Real-time Face Detection**
  - Multiple facial expression recognition
  - Support for 13 different expressions
  - Confidence score display

- **Advanced Hand Detection**
  - Real-time hand gesture recognition
  - Precise finger counting (0-5 fingers)
  - Hand landmark tracking
  - Support for 14 different gestures

- **Enhanced UI**
  - Multiple display modes (default, debug, minimal)
  - Real-time performance metrics
  - Confidence visualization
  - History tracking

- **Performance Optimizations**
  - Multi-threading support
  - Motion detection
  - Frame downscaling
  - Optimized detection parameters

## 🛠️ Technology Stack

- OpenCV - Image processing and webcam handling
- MediaPipe - Hand landmark detection
- Scikit-learn - Machine learning models
- NumPy - Numerical processing
- Scikit-image - Feature extraction

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/expression-gesture-recognition.git
   cd expression-gesture-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Application
```bash
python main.py
```

Options:
- `--use_advanced_features`: Enable advanced feature extraction
- `--use_threading`: Enable multi-threaded processing
- `--use_motion_detection`: Enable motion detection
- `--downscale_factor`: Frame downscale factor (percentage)
- `--ui_mode`: UI mode (default, debug, minimal)

### Finger Counting Application
```bash
python finger_counting_app.py [--threshold 60] [--blur 41]
```

### Advanced Gesture Recognition
```bash
python advanced_gesture_recognition.py [options]
```

## 🎯 Supported Recognition

### Facial Expressions(still working on facial expressions)
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

## 📊 Data Collection & Training

### Collect Training Data
```bash
python collect_training_data.py --data_type [expression|gesture] --output_dir data --num_samples [count]
```

### Train Models
```bash
python train_model.py --model_type [expression|gesture] --data_path [path] --output_path [path]
```

### Authours
- Abhishek(https://github.com/Abhishekmystic-KS/)
- Akshatha(https://github.com/AkshathaaRk/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community
- MediaPipe team
- Contributors and testers
