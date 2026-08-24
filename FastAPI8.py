from fastapi import FastAPI,HTTPException,status
from pydantic import BaseModel,Field
from typing import List,Optional
from enum import Enum,IntEnum
import joblib
import numpy as np
app=FastAPI(description="Testing my Knowledge in FastAPI")
model=joblib.load("model titanic.pkl")
class person(BaseModel):
    name:str=Field(min_length=1,max_length=25)
    prename:str=Field(min_length=1,max_length=15)
    age:int=Field(gt=5,lt=101)
class person_response(person):
    pass
@app.get("/home/{age}",response_model=person_response)
def home(name:str,prename:str,age:int):
    return person(name=name,prename=prename,age=age)
class person_profile(person):
    github_username:str=Field(min_length=6,max_length=100)
class person_profile_response(person):
    pass
@app.get("/home/profile/{name}/{prename}",response_model=person_profile_response)
def render(name:str,prename:str,age:int,github_username:str):
    return person_profile(name=name,prename=prename,age=age,github_username=github_username)
@app.post("/home/profile/add/{name}/{prename}/{age}",response_model=person_profile_response)
def add(name:str,prename:str,age:int,Github_username:str):
    return person_profile(name=name,prename=prename,age=age,github_username=Github_username)
class type_enum(IntEnum):
    first=1
    second=2
    third=3
class modify_person(BaseModel):
    username:Optional[str]=Field("unknown",min_length=5,max_length=30)
    Github_link:Optional[str]=Field("ungiven",min_length=5,max_length=100)
    type:Optional[type_enum]
class modify_person_response(modify_person):
    pass
@app.put("/home/profile/modify/{username}",response_model=modify_person_response)
def modify(username:str,Github_link:str,type:type_enum):
    return  modify_person(username=username,Github_link=Github_link,type=type)
class TitanicFeatures(BaseModel):
    sex_female: float
    sex_male: float
    embarked_C: float
    embarked_Q: float
    embarked_S: float
    deck_A: float
    deck_B: float
    deck_C: float
    deck_D: float
    deck_E: float
    deck_F: float
    deck_G: float
    deck_T: float
    title_Capt: float
    title_Col: float
    title_Don: float
    title_Dr: float
    title_Jonkheer: float
    title_Lady: float
    title_Major: float
    title_Master: float
    title_Miss: float
    title_Mlle: float
    title_Mme: float
    title_Mr: float
    title_Mrs: float
    title_Ms: float
    title_Rev: float
    title_Sir: float
    pclass: float
    age: float
    sibsp: float
    parch: float
    fare: float
    familySize: float
    ticket_number: float
class SurvivalPrediction(str,Enum):
    did_not="DID NOT SURVIVE"
    did= "SURVIVED" 
@app.post("/home/predict",response_model=SurvivalPrediction)
def predict(data:TitanicFeatures):
    features = np.array([[
    data.sex_female,
    data.sex_male,
    data.embarked_C,
    data.embarked_Q,
    data.embarked_S,
    data.deck_A,
    data.deck_B,
    data.deck_C,
    data.deck_D,
    data.deck_E,
    data.deck_F,
    data.deck_G,
    data.deck_T,
    data.title_Capt,
    data.title_Col,
    data.title_Don,
    data.title_Dr,
    data.title_Jonkheer,
    data.title_Lady,
    data.title_Major,
    data.title_Master,
    data.title_Miss,
    data.title_Mlle,
    data.title_Mme,
    data.title_Mr,
    data.title_Mrs,
    data.title_Ms,
    data.title_Rev,
    data.title_Sir,
    data.pclass,
    data.age,
    data.sibsp,
    data.parch,
    data.fare,
    data.familySize,
    data.ticket_number
    ]])
    predictions=(model.predict(features))
    result="DID NOT SURVIVE" if predictions[0]==0 else "SURVIVED"
    return result