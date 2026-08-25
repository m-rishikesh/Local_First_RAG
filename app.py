from rag_main import RAGsys
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("starting RAG server and Watching Files....")
    app.state.rag = RAGsys()
    print("RAG Object Created!!")
    yield
    print("Shutting Down the server")
    app.state.rag.stop()
    print("rag object closed!")



app = FastAPI(lifespan=lifespan)

class SearchQuery(BaseModel):
    query:str
    topic:str | None=None
    top_k:int = 5

@app.post("/search")
def search(query:SearchQuery):
    results = app.state.rag.search(
        query.query,
        query.topic,
        query.top_k
    )

    return {
        "query": query.query,
        "results":results
    }

@app.get("/stats")
def stats():
    return app.state.rag.stats()
