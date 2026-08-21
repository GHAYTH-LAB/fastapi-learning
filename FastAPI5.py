from fastapi import FastAPI
from pydantic import BaseModel,Field
from sklearn.ensemble import RandomForestClassifier
app=FastAPI(title="Test API")
@app.get("/")
def home():
    return {"Message":"welcome to my website"}
@app.get("/Introduction")
def get_introduction(name:str,age:int,uni:str):
    return {"name":name
            ,"age":age
            ,"uni":uni}
model=RandomForestClassifier(random_state=42)
class modelscheme(BaseModel):
    name:str=Field(max_length=25,min_length=5)
    age:int=Field(gt=10,lt=25)
    uni:str=Field(max_length=50,min_length=3)
@app.post("/estimate/{age}/{uni}")
def predict(data:modelscheme):
    features=[[
        data.name
        ,data.age
        ,data.uni
    ]]
    predictions=model.predict(features)
    return {"predictions":predictions[0]}
@app.get("/Github/{name}")
def render(name:str,Github_link:str):
    return{"name":name
           ,"Github_link":Github_link}
@app.put("/Github/modify/{username}")
def change(username:str,new_username):
    return{"name":new_username
           ,"old name":username}
@app.patch("/FatApi/{day}")
def change_partially(day:int):
    return{"old date":day
           ,"day":8}
