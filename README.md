# Cat vs. Dog Classification

This project implements a Convolutional Neural Network (CNN) for classifying images of cats and dogs using PyTorch. It includes data preprocessing, augmentation, model training, evaluation, and prediction functionalities.

## Requirements

To reproduce this project, you'll need to set up the environment with the following dependencies:

- Python >= 3.7
- PyTorch >= 1.10
- torchvision
- matplotlib
- numpy
- pandas
- tqdm
- scikit-learn
- Pillow

You can create a virtual environment and install the dependencies using `pip`:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Activate the virtual environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
# Data Prepairtation
Run the following commands in your project directory to automate the setup:
```bash
!mkdir data

!curl -L -o /data/cat-and-dog.zip\
https://www.kaggle.com/api/v1/datasets/download/tongpython/cat-and-dog

!unzip -q /data/cat-and-dog.zip -d /content/data

!rm -rf /data/cat-and-dog.zip
```
# Usage
Once the data is prepared, you can train the model using the following command:
```bash
python3 train.py
```
To evaluate the trained model on test set, run:
```bash
python3 evaluate.py
```
To make predictions on new images, run:
```bash
streamlit run app.py
```


# Running on Colab
* All training, testing, and prediction tasks are implemented in the Code_test.ipynb file.
