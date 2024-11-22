import uvicorn


def start_api():
    """
    Entry point to start the FastAPI application using Uvicorn.
    """
    uvicorn.run("panel4.Panel4-api:app", host="0.0.0.0", port=8000, reload=True)
