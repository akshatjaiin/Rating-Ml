import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load and prepare the dataset
dataset = pd.read_csv("Dataset.csv")
dataset = dataset.dropna(axis=0)

# Separate target variable and features
y = dataset['Aggregate rating']
features = ['Restaurant ID', 'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 'Currency', 
            'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 
            'Price range', 'Votes']
X = dataset[features]

# Define categorical and numerical features
categorical_features = ['Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
numerical_features = ['Restaurant ID', 'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 
                      'Price range', 'Votes']

# Create preprocessing pipelines for both categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create a pipeline that combines preprocessing with the Decision Tree model
def create_pipeline(max_leaf_nodes):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0))
    ])
    return model

# Function to calculate MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = create_pipeline(max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Evaluate model performance with different max_leaf_nodes
best_max_leaf_nodes = None
best_mae = float('inf')

for max_leaf_nodes in [40, 500, 1000, 3000, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"MAE with max_leaf_nodes={max_leaf_nodes}: {mae}")
    
    if mae < best_mae:
        best_mae = mae
        best_max_leaf_nodes = max_leaf_nodes

# Fit the final model with the best max_leaf_nodes
final_model = create_pipeline(best_max_leaf_nodes)
final_model.fit(X, y)
