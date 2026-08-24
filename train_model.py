import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer,LabelEncoder
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score
import joblib
# data handling and cleaning
df=pd.read_csv("train_titanic_classification.csv")
df1=pd.read_csv("test_titanic_classification.csv")
#Data cleaning
df.columns=(df.columns
            .str.lower()
            .str.strip())
extracted_id=df1["PassengerId"]
df["familySize"] = df["sibsp"] + df["parch"] + 1
df["deck"]=df["cabin"].str[0]
df["title"] = df["name"].str.extract(r',\s*([A-Za-z]+)\.')
age_fill=df.groupby(["sex"])["age"].median()
deck_fill=df["deck"].mode()[0]
embarked_fill=df["embarked"].mode()[0]
df=df.fillna({
    "age":df["sex"].map(age_fill)
    ,"deck":deck_fill
    ,"embarked":embarked_fill
    ,
})
df["ticket_number"] = df["ticket"].str.extract(r'(\d+)')
df=df.drop(columns=["cabin","name","ticket"])
df=df.dropna(subset=["title","ticket_number"])
df["ticket_number"]=df["ticket_number"].astype(int)
#data cleaning for test dataset
df1.columns=(df1.columns
            .str.lower()
            .str.strip())
df1["familySize"] = df1["sibsp"] + df1["parch"] + 1
df1["deck"]=df1["cabin"].str[0]
df1["title"] = df1["name"].str.extract(r',\s*([A-Za-z]+)\.')
df1=df1.fillna({
    "age":df1["sex"].map(age_fill)
    ,"deck":deck_fill
    ,"embarked":embarked_fill
})
df1["ticket_number"] = df1["ticket"].str.extract(r'(\d+)')
df1=df1.drop(columns=["cabin","name","ticket"])
df1["ticket_number"]=df1["ticket_number"].astype(int)
df1=df1.fillna({
    "fare":df["fare"].median()
})
y_train=df["survived"]
X_train=df.drop(columns=["survived","passengerid"])
print(X_train.info())
X_test=df1.drop(columns="passengerid")
columns_cat=X_train.select_dtypes(include=["object","str"]).columns
columns_num=X_train.select_dtypes(exclude=["object","str"]).columns
encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
X_train_cat=encoder.fit_transform(X_train[columns_cat])
X_test_cat=encoder.transform(X_test[columns_cat])
# Get the names of the features created by OneHotEncoder
cat_feature_names=encoder.get_feature_names_out(columns_cat)
scaler=QuantileTransformer(n_quantiles=500)
X_train_num=scaler.fit_transform(X_train[columns_num])
X_test_num=scaler.transform(X_test[columns_num])
# Save the numerical feature names
num_feature_names=columns_num.tolist()
X_train=np.hstack([X_train_cat,X_train_num])
X_test=np.hstack([X_test_cat,X_test_num])
# Get the final feature names
feature_names=list(cat_feature_names)+num_feature_names
print("Number of final features:", len(feature_names))
print("Final features:")
print(feature_names)
# See the actual final X_train with column names
X_train_final=pd.DataFrame(X_train,columns=feature_names)
print(X_train_final.head())
voting=VotingClassifier(
    estimators=[
        ("rf",RandomForestClassifier(random_state=42))
        ,("knn",KNeighborsClassifier())
        ,("xg",XGBClassifier(random_state=42))
    ]
)
grid=GridSearchCV(
    estimator=voting
    ,param_grid={
    "rf__n_estimators":[150,250],
    "rf__max_depth":[5,10],
    "rf__min_samples_leaf":[5,10],
    "knn__n_neighbors":[7],
    "xg__n_estimators":[200],
    "xg__learning_rate":[0.1]
    }
    ,cv=5
    ,n_jobs=-1
)
grid.fit(X_train,y_train)
joblib.dump(grid.best_estimator_,"model titanic.pkl")
