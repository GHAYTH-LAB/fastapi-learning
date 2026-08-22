from fastapi import FastAPI,HTTPException
from catboost import CatBoostClassifier
from enum import IntEnum
from typing import List,Optional
from pydantic import BaseModel,Field
app=FastAPI(description="day6")
model=CatBoostClassifier(random_state=42)
@app.get("/")
def home():
    return{
        "name":"Ghayth"
        ,"Age":21
    }
@app.get("/Github/{username}")
def github(username:str,user_id:int,Github_link:str):
    return{
        "username":username
        ,"user_id":user_id
        ,"Github_link":Github_link
    }
class priorityc(IntEnum):
    Low=1
    Medium=2
    High=3
class userschema(BaseModel):
    age:int=Field(ge=12,lt=101)
    salary:Optional[int]=Field(None,gt=100,lt=10000,description="salary_value")
    priority:Optional[priorityc]=None
@app.post("/model/predict")
def predict(data:userschema):
    features=[[
        data.age
        ,data.salary
        ,data.priority
    ]]
    predictions=model.predict(features)
    return{"predictions":predictions[0]}