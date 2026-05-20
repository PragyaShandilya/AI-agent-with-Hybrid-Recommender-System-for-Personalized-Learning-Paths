from fastapi import FastAPI
from pydantic import BaseModel
from Hybrid import hybrid_recommend
from fastapi.responses import RedirectResponse

app = FastAPI()

class UserQuery(BaseModel):
    skills: str
    level: str

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/recommend")
def recommend(query: UserQuery):

    results = hybrid_recommend(query.skills, query.level)

    return results.to_dict(orient="records")