import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle

# reading cleaned data
df_x=pd.read_csv('cleaned_data_x.csv')
df_y=pd.read_csv('cleaned_data_y.csv')

# splitting the x and y labels
features = ['national_inv', 'lead_time', 'sales_1_month', 'pieces_past_due', 'perf_6_month_avg',
            'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
X = df_x[features]
y = df_y['went_on_backorder']

# splitting Data into train and test.
X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=20, random_state=42)

# Checking shape.
print (X_train.shape, y_train.shape)
print (X_valid.shape, y_valid.shape)


# making Model.
model=RandomForestClassifier()
model.fit(X_train,y_train)
# accuracy of RandomForest Model
y_predxgb = model.predict(X_valid)
report2 = classification_report(y_valid, y_predxgb)
print(report2)
print("Accuracy of the RandomForest Model is:",accuracy_score(y_valid,y_predxgb)*100,"%")

# dumping model
filename='PredictModel.pickle'
pickle.dump(model,open(filename,'wb'))