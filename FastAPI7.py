from fastapi import FastAPI, HTTPException
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Trying my first ML model API", description="Testing what I learned this week in my FastAPI zero to hero journey")

class HobbyEnum(str, Enum):
    football = "Football"
    handball = "Handball"
    leetcode = "Leetcode"
class Person(BaseModel):
    name: str = Field(min_length=1, max_length=15)
    prename: str = Field(min_length=1, max_length=20)
    age: int = Field(gt=5, lt=100)
    hobby_name: HobbyEnum
class PersonResponse(Person):
    pass
@app.get("/", response_model=PersonResponse)
def home():
    try:
        return Person(name="Ghayth", prename="Abidli", age=20, hobby_name=HobbyEnum.leetcode)
    except Exception:
        raise HTTPException(status_code=400, detail="The request is invalid")
class Github(BaseModel):
    username: str
    Github_link: str = Field(min_length=10)
class GithubResponse(Github):
    pass
@app.get("/Github/{username}/{Github_link:path}", response_model=GithubResponse)
def render(username: str, Github_link: str):
    return Github(username=username, Github_link=Github_link)
class NewData(Github):
    id: int = Field(gt=10000, lt=100000)
class NewDataResponse(NewData):
    pass
@app.post("/Github/{new_username}/{id}", response_model=NewDataResponse)
def add(new_username: str, id: int, Github_link: str):
    return NewDataResponse(username=new_username, Github_link=Github_link, id=id)
class ChangePerson(BaseModel):
    name: Optional[str] = Field(default="Ghayth", min_length=3)
    age: Optional[int] = Field(default=None, gt=5, lt=21)
    prename: str = Field(min_length=1)
class ChangePersonResponse(ChangePerson):
    pass
@app.put("/Github/change/{name}/{age}", response_model=ChangePersonResponse)
def change(name: str, age: int, prename: str):
    return ChangePersonResponse(name=name, age=age, prename=prename)
class ChangePatchPerson(BaseModel):
    name: Optional[str] = Field(default="Ghayth", min_length=4)
    age: Optional[int] = Field(default=None, gt=5)
class ChangePatchPersonResponse(ChangePatchPerson):
    pass
@app.patch("/Github/change/partially", response_model=ChangePatchPersonResponse)
def modify(new_name: str, new_age: int):
    return ChangePatchPersonResponse(name=new_name, age=new_age)
model = joblib.load(r"C:\Users\abidli\Desktop\FastAPI\breast_cancer.pkl")
class CancerFeatures(BaseModel):
    mean_radius: float = Field(..., description="Mean radius")
    mean_texture: float = Field(..., description="Mean texture")
    mean_perimeter: float = Field(..., description="Mean perimeter")
    mean_area: float = Field(..., description="Mean area")
    mean_smoothness: float = Field(..., description="Mean smoothness")
    mean_compactness: float = Field(..., description="Mean compactness")
    mean_concavity: float = Field(..., description="Mean concavity")
    mean_concave_points: float = Field(..., description="Mean concave points")
    mean_symmetry: float = Field(..., description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., description="Mean fractal dimension")
    radius_error: float = Field(..., description="Radius error")
    texture_error: float = Field(..., description="Texture error")
    perimeter_error: float = Field(..., description="Perimeter error")
    area_error: float = Field(..., description="Area error")
    smoothness_error: float = Field(..., description="Smoothness error")
    compactness_error: float = Field(..., description="Compactness error")
    concavity_error: float = Field(..., description="Concavity error")
    concave_points_error: float = Field(..., description="Concave points error")
    symmetry_error: float = Field(..., description="Symmetry error")
    fractal_dimension_error: float = Field(..., description="Fractal dimension error")
    worst_radius: float = Field(..., description="Worst radius")
    worst_texture: float = Field(..., description="Worst texture")
    worst_perimeter: float = Field(..., description="Worst perimeter")
    worst_area: float = Field(..., description="Worst area")
    worst_smoothness: float = Field(..., description="Worst smoothness")
    worst_compactness: float = Field(..., description="Worst compactness")
    worst_concavity: float = Field(..., description="Worst concavity")
    worst_concave_points: float = Field(..., description="Worst concave points")
    worst_symmetry: float = Field(..., description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., description="Worst fractal dimension")
class ResponseEnum(str, Enum):
    benign = "benign"
    malignant = "malignant"
class PredictionResponse(BaseModel):
    response: ResponseEnum
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CancerFeatures):
    features = np.array([[data.mean_radius, data.mean_texture, data.mean_perimeter, data.mean_area, data.mean_smoothness, data.mean_compactness, data.mean_concavity, data.mean_concave_points, data.mean_symmetry, data.mean_fractal_dimension, data.radius_error, data.texture_error, data.perimeter_error, data.area_error, data.smoothness_error, data.compactness_error, data.concavity_error, data.concave_points_error, data.symmetry_error, data.fractal_dimension_error, data.worst_radius, data.worst_texture, data.worst_perimeter, data.worst_area, data.worst_smoothness, data.worst_compactness, data.worst_concavity, data.worst_concave_points, data.worst_symmetry, data.worst_fractal_dimension]])
    prediction = model.predict(features)
    result = ResponseEnum.benign if prediction[0] == 1 else ResponseEnum.malignant
    return PredictionResponse(response=result)
