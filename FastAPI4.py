from fastapi import FastAPI
from pydantic import BaseModel,Field
from sklearn.ensemble import RandomForestClassifier
app=FastAPI()
@app.get("/")
def home():
    return "welcome home"
class userschema(BaseModel):
    name:str=Field(min_length=3,max_length=20)
    prename:str=Field(min_length=5,max_length=25)
    age:int=Field(gt=5,lt=21)
    salary:int=Field(gt=6,lt=22)
model=RandomForestClassifier(random_state=42)
@app.post("/predict")
def predict(data:userschema):
    features=[[
        data.name
        ,data.prename
        ,data.age
        ,data.salary
    ]]
    predictions=model.predict(features)
    return{
        "predictions":predictions[0]
    }