# Image Classification with Transfer Learning

## Overview

This project involves building an image classification model using transfer learning. The goal is to classify images of cats and dogs into their respective categories. We use the MobileNetV2 architecture for transfer learning, fine-tuned to improve performance on our specific dataset.

## Project Structure

ImageClassification-TransferLearning/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── src/
│   ├── build_model.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   └── train_model.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── venv/


## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Keras
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amitjadhav055/ImageClassification-TransferLearning.git
   ```

2. Navigate to the project directory:

   cd ImageClassification-TransferLearning


3. Create and activate a virtual environment:
   
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`


4. Install the required packages:
   
   pip install -r requirements.txt


### Dataset

The dataset consists of images of cats and dogs, divided into training and testing sets. Each set contains images in the following structure:
```
data/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

### Usage

1. **Data Preprocessing:**
   Run the following script to preprocess the data:

   python src/data_preprocessing.py


2. **Build the Model:**
   Create and save the transfer learning model:

   python src/build_model.py


3. **Train the Model:**
   Train the model using the training data:

   python src/train_model.py


4. **Evaluate the Model:**
   Evaluate the trained model on the test set and generate a classification report:

   python src/evaluate_model.py


### Results

After training and evaluating the model, you should get a classification report and a confusion matrix displaying the model's performance on the test data.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

- TensorFlow and Keras for the machine learning framework.
- MobileNetV2 for the pre-trained model.

---

Feel free to modify any sections or add more details as needed!