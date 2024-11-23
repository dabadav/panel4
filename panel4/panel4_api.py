"""FastAPI File
"""

import csv
import os
from typing import List
import logging
import time
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.logger import logger
import pandas as pd
from panel4.panel4 import RecSys, InMemorySimilarityEngine, Visitor, CorpusManager, spacy_embedding


# ------------------ Logger Configration ------------------
# Generate unique file names with timestamps
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# FastAPI Logs
file_handler = logging.FileHandler(f"recsys_api_{now}.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# RecSys Logs
EVENT_LOG_FILE = f"visitor_events_{now}.csv"
RECOMMENDATION_LOG_FILE = f"recommendations_{now}.csv"


def log_to_csv(file_path: str, **kwargs):
    """
    General function to log data to a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        **kwargs: Key-value pairs to log as CSV columns.
    """
    try:
        # Check if the file exists to determine if headers are needed
        file_exists = os.path.exists(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as log_file:
            csv_writer = csv.DictWriter(log_file, fieldnames=kwargs.keys())
            if not file_exists or os.stat(file_path).st_size == 0:
                csv_writer.writeheader()  # Write headers only if the file is new
            csv_writer.writerow(kwargs)
    except Exception as e:
        logger.error("Failed to log to %s: %s", file_path, e)


# ------------------ RecSys FastAPI ------------------
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


@app.post("/visitor/{visitor_id}/update/")
async def update_visitor(visitor_id: str, update_input: VisitorUpdateInput):
    """Add visitor events"""
    visitor = visitors.get(visitor_id, Visitor())
    content = recsys.contents.get(update_input.content_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Content not found.")
    visitor.update(content, update_input.event, update_input.timestamp, update_input.isrecommended)
    visitors[visitor_id] = visitor

    log_to_csv(
        EVENT_LOG_FILE,
        visitor_id=visitor_id,
        content_id=update_input.content_id,
        event=update_input.event,
        timestamp=update_input.timestamp,
        isrecommended=update_input.isrecommended,
    )

    return {"message": "Visitor history updated successfully."}


@app.get("/visitor/{visitor_id}/recommendations/", response_model=RecommendationOutput)
async def get_recommendations(visitor_id: str, num_recommendations: int = 12):
    """Given visitor content navigation history provide content recommendations"""
    visitor = visitors.get(visitor_id)
    if not visitor:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    try:
        recommended_ids = recsys.recommend(visitor, num_recommendations)

        log_to_csv(
            RECOMMENDATION_LOG_FILE,
            visitor_id=visitor_id,
            recommended_ids=recommended_ids,
            timestamp=time.time(),
        )

        return {"recommended_ids": recommended_ids}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
