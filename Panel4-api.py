from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import time
from Panel4b import RecSys, InMemorySimilarityEngine, Visitor, CorpusManager, spacy_embedding

class ContentInput(BaseModel):
    id: int
    text: str

class VisitorUpdateInput(BaseModel):
    content_id: int
    event: str
    timestamp: float = time.time()
    isrecommended: int

class RecommendationOutput(BaseModel):
    recommended_ids: List[int]

app = FastAPI()

# Global instances
merged_df = pd.read_csv("text_corpus.csv", index_col=0)
corpus_manager = CorpusManager(spacy_embedding)
corpus = corpus_manager.ingest_corpus(merged_df)
similarity_engine = InMemorySimilarityEngine()
recsys = RecSys(similarity_engine, corpus)
visitors = {}

# @app.post("/content/")
# async def add_content(content_input: ContentInput):
#     corpus_manager.add_content(content_input)
#     return {"message": "Content added successfully."}

@app.post("/visitor/{visitor_id}/update/")
async def update_visitor(visitor_id: str, update_input: VisitorUpdateInput):
    visitor = visitors.get(visitor_id, Visitor())
    content = recsys.contents.get(update_input.content_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Content not found.")
    visitor.update(content, update_input.event, update_input.timestamp, update_input.isrecommended)
    visitors[visitor_id] = visitor
    return {"message": "Visitor history updated successfully."}

@app.get("/visitor/{visitor_id}/recommendations/", response_model=RecommendationOutput)
async def get_recommendations(visitor_id: str, num_recommendations: int = 12):
    visitor = visitors.get(visitor_id)
    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    try:
        recommended_ids = recsys.recommend(visitor, num_recommendations)
        return {"recommended_ids": recommended_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
