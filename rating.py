import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Dataset.csv")
dataset = dataset.dropna(axis=0)
print(dataset.columns)
y = dataset['Aggregate rating']
features = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address',
       'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines',  
       'Average Cost for two', 'Currency', 'Has Table booking',
       'Has Online delivery', 'Is delivering now', 'Switch to order menu',   
       'Price range', 'Rating color', 'Rating text',     
       'Votes']
X = dataset[features]



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# spliting data to train and validate
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
print(train_X.columns)
print(val_X.columns)
for max_leaf_nodes in [40, 500, 1000, 3000, 5000]:
   print(f"mae: {get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)}")
