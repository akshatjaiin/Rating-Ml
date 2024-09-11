# Restaurant Rating Prediction using Decision Tree Regression

This project implements a decision tree regression model to predict the aggregate rating of restaurants based on various features such as location, price range, delivery options, and more. The model uses a combination of numerical and categorical data, which are preprocessed and fed into a pipeline that includes data scaling, one-hot encoding, and decision tree regression.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [License](#license)

## Installation

To set up this project, clone the repository and install the required Python packages by running:

```bash
pip install -r requirements.txt
Required Libraries
pandas: For data handling and manipulation.
```
Features
Data Preprocessing: The script preprocesses both numerical and categorical features. Numerical features are standardized using StandardScaler, while categorical features are encoded using OneHotEncoder.
Model Training: A decision tree regression model is trained to predict the aggregate rating of restaurants based on the provided features.
Hyperparameter Tuning: The maximum number of leaf nodes in the decision tree is tuned to find the best-performing model based on the lowest mean absolute error (MAE).
Pipeline: The preprocessing steps and model training are combined into a single Pipeline for streamlined execution.
Usage
Dataset Preparation
Ensure the dataset file (Dataset.csv) is available in the project directory.
The dataset should have the following columns:
Aggregate rating: The target variable, representing the restaurant rating.
Restaurant ID: A unique identifier for each restaurant.
Country Code: The country code for the restaurant's location.
Longitude, Latitude: The geographical coordinates of the restaurant.
Average Cost for two: The average cost for two people dining at the restaurant.
Currency: The currency used at the restaurant.
Has Table booking, Has Online delivery, Is delivering now, Switch to order menu: Categorical features indicating the availability of certain services.
Price range: The price range category of the restaurant.
Votes: The number of votes or reviews the restaurant has received.
Model Training
Run the script: The model will automatically preprocess the data, split it into training and validation sets, and train the model.

Hyperparameter Tuning: The model will test different values for max_leaf_nodes (40, 500, 1000, 3000, 5000) and report the Mean Absolute Error (MAE) for each configuration.

python rating.py
Sample output:


MAE with max_leaf_nodes=40: 0.532
MAE with max_leaf_nodes=500: 0.428
MAE with max_leaf_nodes=1000: 0.422
MAE with max_leaf_nodes=3000: 0.420
MAE with max_leaf_nodes=5000: 0.425
The best configuration will be selected automatically based on the lowest MAE.

Model Evaluation
After training, the model will be fitted on the entire dataset using the best max_leaf_nodes value, and it can be used for further prediction tasks.

Project Structure
├── Dataset.csv                 # Input dataset
├── rating.py            # Main script for data preprocessing, model training, and evaluation
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation

### Notes:
1. You need to create a `requirements.txt` file with the dependencies listed under "Required Libraries."
2. Ensure the dataset (`Dataset.csv`) is available in the same directory as the script. The dataset must be preprocessed to remove any missing values (`dropna` is used in the script).
3. The script uses `train_test_split` to split the data into training and validation sets (80% training, 20% validation).
![image](https://github.com/user-attachments/assets/c508bb1a-ca94-457c-a1d2-85c4f15b831b)
