from fastapi import FastAPI
app=FastAPI()
@app.get("/")
def home():
    return {"Name":"Ghayth"
            ,"Prename":"Abidli"
            ,"Age":20
            ,"Hobby":"Cp"}
@app.post("/user/{username}")
def change(username:int):
    return {"user":username}
@app.get("/Github")
def render():
    return {"Name":"Ghayth Abidli"
            ,"Github link":"https://github.com/GHAYTH-LAB"}
@app.post("/Github/{username}")
def change(age:int,username:str):
    return {"Username":username
            ,"age":age}
@app.put("/Github/{username}/{id}")
def change(id:int,username:str):
    return {"username":username
            ,"id":id}
@app.patch("/home/{uni}")
def change_partially(uni:str):
    return {"name":"Abidli"
            ,"uni":uni
            ,"id":253024}
@app.delete("/home/delete")
def delete():
    return{"Name":"unkown"
           ,"passed":True}
