from fastapi import FastAPI
app=FastAPI()
database=[
    {"user_id":1,"user_name":"Ghayth","age":20,"phone_number":55501390}
    ,{"user_id":2,"user_name":"Xavi","age":44,"phone_number":55501380}
    ,{"user_id":3,"user_name":"Messi","age":39,"phone_number":55501370}
    ,{"user_id":4,"user_name":"Cristiano","age":42,"phone_number":55501360}
    ,{"user_id":5,"user_name":"Neymar","age":34,"phone_number":55501350}
]
@app.get("/dashboard")
async def help(user:str,problem_case:int,phone_number:int):
    return {"name":user
            ,"phone_number":phone_number
            ,"problem":problem_case}
@app.get("/")
def add_user(user_name:str,user_id:int):
    return {"name":user_name
            ,"user_id":user_id}
@app.get("/to do/{todo_id}")
def get_user(todo_id:int):
    for user in database:
        if user["user_id"]==todo_id:
            return {"result":user}
