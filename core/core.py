import numpy as np
from core.models.MathModel import MathModel
from core.models.NeuralNetModel import NeuralNetModel
from core.optimisers.FletcherReevesOptimiser import FletcherReevesOptimiser
from core.optimisers.BoltzmannOptimiser import BoltzmannOptimiser


def model(model_name, model_params):
    if model_name == "NeuralNetModel":
        model = NeuralNetModel()
    if model_name == "MathModel":
        model = MathModel()
    x = [
        model_params['cNaCl'],
        model_params['cHCl'],
        model_params['cNaOH'],
        model_params['DBL'],
    ]
    y, log = model.calculate(x)
    response = {
        "model_params": x,
        "model_result": y.tolist(),
    }
    if log is not None:
        response["model_log"] = log.tolist()
    return response


def optimize(optimiser_name, model_name, optimiser_params):
    if optimiser_name == "BoltzmannOptimiser":
        optimiser_class = BoltzmannOptimiser
    if optimiser_name == "FletcherReevesOptimiser":
        optimiser_class = FletcherReevesOptimiser

    if model_name == "NeuralNetModel":
        model = NeuralNetModel()
    if model_name == "MathModel":
        model = MathModel()

    optimiser = optimiser_class(
        optimiser_params['cNaCl'],
        optimiser_params['cHClMin'],
        optimiser_params['cHClMax'],
        optimiser_params['cNaOHMin'],
        optimiser_params['cNaOHMax'],
        optimiser_params['DBLMin'],
        optimiser_params['DBLMax'],
        model
    )
    yMin, xMin = optimiser.optimize()

    response = {"optimised_params": xMin.tolist(), "optimised_result": yMin.tolist()}

    math_model = MathModel()
    yMin, log = math_model.calculate(xMin)
    response["math_model_result"] = yMin.tolist()
    response["math_model_log"] = log.tolist()

    return response

