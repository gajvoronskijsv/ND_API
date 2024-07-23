from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from core.core import model, optimize

app = FastAPI()


class ModelName(str, Enum):
    MathModel = "MathModel"
    NeuralNetModel = "NeuralNetModel"


class ModelParams(BaseModel):
    cNaCl: float = 0.15
    cHCl: float = 0.3
    cNaOH: float = 0.3
    DBL: float = 50


@app.post("/model/{model_name}")
async def post_model(model_name: ModelName, model_params: ModelParams):
    return model(model_name, model_params.dict())


class OptimiserName(str, Enum):
    BoltzmannOptimiser = "BoltzmannOptimiser"
    FletcherReevesOptimiser = "FletcherReevesOptimiser"


class OptimiserParams(BaseModel):
    cNaCl: float = 0.15
    cHClMin: float = 0.3
    cHClMax: float = 1
    cNaOHMin: float = 0.3
    cNaOHMax: float = 1
    DBLMin: float = 50
    DBLMax: float = 150


@app.post("/optimiser/{optimiser_name}/model/{model_name}")
async def post_optimiser(optimiser_name: OptimiserName, model_name: ModelName, optimiser_params: OptimiserParams):
    return optimize(optimiser_name, model_name, optimiser_params.dict())
