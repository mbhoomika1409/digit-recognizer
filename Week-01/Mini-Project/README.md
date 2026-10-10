# ✍️ Digit Recognizer: MNIST CNN with a Live Drawing Canvas

A convolutional neural network trained on MNIST, served through a Flask web app. Draw a digit on the canvas and the model predicts it in real time, with a confidence bar for every digit.

**Built by:** Bhoomika M T
**Project:** Deep Learning & AI Internship, Week 1 Mini Project (NEXIS Solutions)

## Demo

![Digit Recognizer demo](images/demo1.png)

## Results

| Metric | Value |
|--------|-------|
| **Test accuracy** | **99.20%** |
| Best validation accuracy | 99.13% |
| Parameters | 421,834 |

Every digit class has precision and recall above 98.4%.

### Training curves
![Training curves](images/training_curves.png)

### Confusion matrix (10,000 test images)
![Confusion matrix](images/confusion_matrix.png)

### Misclassified examples
![Misclassified](images/misclassified.png)

## How it works

1. **Model:** a CNN with 2 convolution blocks (Conv, BatchNorm, ReLU, MaxPool), then dropout and 2 fully connected layers.
2. **Training:** 10 epochs, Adam optimizer (lr = 0.001), batch size 128, cross-entropy loss.
3. **Data split:** 54,000 train / 6,000 validation / 10,000 test. The test set is only used once, at the end.
4. **Augmentation:** random rotation, shift and zoom on training images, so the model handles messy hand-drawn digits.
5. **Preprocessing for the canvas:** the drawing is cropped, scaled to 20x20, centered in 28x28 by its center of mass, and normalized exactly like the training data.
6. **Web app:** Flask serves the page and a `/predict` endpoint. The canvas sends the drawing as base64, and the server returns the digit, confidence and all 10 probabilities.

## Project structure

```
Mini-Project/
├── images/              # screenshot and result plots
├── templates/
│   └── index.html       # drawing canvas UI
├── app.py               # Flask backend + preprocessing
├── model.py             # CNN architecture
├── train.py             # training, validation, loss curves
├── evaluate.py          # test accuracy, confusion matrix
├── digit_cnn.pth        # trained weights
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/mbhoomika1409/Deep-Learning-AI-Internship.git
cd Deep-Learning-AI-Internship/Week-01/Mini-Project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt

python app.py
```

Open **http://127.0.0.1:5001** and draw a digit.

To retrain the model from scratch:

```bash
python train.py       # saves digit_cnn.pth and training_curves.png
python evaluate.py    # test accuracy, confusion matrix, misclassified images
```

## Tech stack

PyTorch, torchvision, Flask, NumPy, SciPy, Pillow, scikit-learn, Matplotlib, HTML/CSS/JavaScript (Canvas API)

## Tips for best predictions

Draw the digit **large and bold**, filling most of the canvas, as a single clean stroke or two.