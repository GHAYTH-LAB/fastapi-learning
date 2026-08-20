from fastapi import FastAPI
api=FastAPI()
@api.get("/")
def index():
    return {"message":"Hello world"}
@api.get("/Github")
def Github():
    return {"name":"Ghayth"
            ,"Link":"https://github.com/GHAYTH-LAB"
            ,"University":"Insat"
            }