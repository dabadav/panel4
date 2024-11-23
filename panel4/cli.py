"""Module to start FastAPI
"""

import argparse
import uvicorn


def start_api():
    """
    Entry point to start the FastAPI application using Uvicorn with configurable settings.
    """
    parser = argparse.ArgumentParser(description="Start the FastAPI application.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the FastAPI application (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the FastAPI application (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=True,
        help="Enable/disable Uvicorn auto-reload (default: True)",
    )

    args = parser.parse_args()

    uvicorn.run("panel4.panel4_api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    start_api()
