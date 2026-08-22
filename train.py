from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
X,y=load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
model=CatBoostClassifier(random_state=42
                         ,iterations=50
                         ,learning_rate=0.1
                         )
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(accuracy_score(predictions,y_test))
joblib.dump(model,"breast_cancer.pkl")
