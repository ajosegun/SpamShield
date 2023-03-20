import numpy as np
import uvicorn
from fastapi import FastAPI, Path

import core

fast_app = FastAPI(
    title="SpamShield",
    description="A simple API that use PySpark's Machine Learning library to predict if an Email/SMS is a spam or not.",
    version="0.1",
)


@fast_app.get("/")
def predict(message: str):
    """
    A simple function that receive a message and returns the SPAM prediction.
    :param question:
    :return: prediciton  0 = Not Spam // 1 = Spam
    """
    
    if message.strip() == "":
       return {"Error", "message cannot be empty"}
    
    result = core.predict(message)
    
    return {"prediciton": result}
    
