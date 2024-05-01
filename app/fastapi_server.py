# Import the necessary modules
from getpass import getpass  # Use getpass to hide the password input
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session
from .admin_database import get_database, verify_hash, Admin
from typing import Optional, Union
import os
import sys

# Define the function to create the FastAPI application
def create_app(secret_key: str):
    # Set the project root path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the project root directory to the Python path
    sys.path.insert(0, project_root)

    # Import routers from the 'routers' package using relative imports
    from .routers import admin, user, login

    # Create FastAPI application object
    app = FastAPI()

    #Allow CORS for only the React frontend server
    # origins = [
    #     "http://85.239.241.96:3000",  # Your React frontend server's HTTP URL
    #     "http://api.bittaudio.ai",
    #     "http://144.91.69.154:8000",
    #     "http://localhost:3000",
    #     "http:127.0.0.1:3000",
    #     "http://89.37.121.214:44107",
    #     "http://149.11.242.18:14428",
    #     "http://bittaudio.ai",
    #     "http://v1.bittaudio.ai",
    #     "http://v2.bittaudio.ai",
    # ]


    # # Allow CORS only if not handled by Nginx
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=origins,
    #     allow_credentials=True,
    #     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    #     allow_headers=["*"],
    #     expose_headers=["*"],
    # )


    # Define the list of allowed origins
    origins = [
        "http://85.239.241.96:3000",  # Your React frontend server's HTTP URL
        "http://api.bittaudio.ai",
        "http://144.91.69.154:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://89.37.121.214:44107",
        "http://149.11.242.18:14428",
        "http://bittaudio.ai",
        "http://v1.bittaudio.ai",
        "http://v2.bittaudio.ai",
        "http://93.114.160.254:40321"
    ]

    # Allow CORS for all origins specified in the list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(login.router, prefix="", tags=["Authentication"])
    app.include_router(admin.router, prefix="", tags=["Admin"])
    app.include_router(user.router, prefix="", tags=["User"])

    return app

