# 🐱 Dog vs Cat Classifier 🐶

A simple yet powerful image classification application that can distinguish between dogs and cats using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## ✨ Features

- 🖼️ Upload and classify images of dogs and cats
- 📊 View prediction confidence with visual indicators
- 🧠 Trained CNN model with good accuracy
- 🖥️ Streamlit web interface for easy interaction
- 📝 Command-line interface for inferencing

## 🛠️ Installation

### Prerequisites

- Python 3.8+ installed
- pip package manager

### Option 1: Standard Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dog-cat-classifier.git
   cd dog-cat-classifier
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Conda (for Apple Silicon M1/M2)

1. Create a conda environment using the provided YAML file:
   ```bash
   conda env create -n dogcat python==3.11 -y
   conda activate dogcat
   ```

2. Install additional requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Then open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

### Command Line Interface

For quick predictions without the web interface:
```bash
python cli.py path/to/your/image.jpg
```

Optional arguments:
- `--model`: Specify a different model file (default: dog_cat_cnn_model.h5)

Example:
```bash
python cli.py samples/my_cat.jpg --model models/custom_model.h5
```

## 🧪 Model Training

The model was trained on a dataset of dog and cat images. The training process is documented in `training_a_cnn_with_custom_dataset_keras.py`.

Key model architecture:
- Input: 128x128 RGB images
- 3 convolutional layers with max pooling
- Dense layers with dropout for regularization
- Binary classification output (dog or cat)

## 📁 Project Structure

```
dog-cat-classifier/
├── streamlit_app.py        # Web interface
├── cli.py        # Command line tool
├── dog_cat_cnn_model.h5    # Trained model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Dataset from Kaggle's Dogs vs Cats competition
- Built with TensorFlow and Streamlit