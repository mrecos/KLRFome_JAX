"""Model serialization utilities."""

import pickle
from typing import Optional
import json

from ..api import KLRfome


def save_model(model: KLRfome, file_path: str):
    """
    Save a fitted KLRfome model to disk.
    
    Parameters:
        model: Fitted KLRfome model
        file_path: Path to save model
    """
    # Save model state (hyperparameters, fitted coefficients, etc.)
    # Note: Training data is not saved by default to reduce file size
    model_state = {
        'sigma': model.sigma,
        'lambda_reg': model.lambda_reg,
        'n_rff_features': model.n_rff_features,
        'window_size': model.window_size,
        'seed': model.seed,
        'alpha': model._fit_result.alpha if model._fit_result else None,
        'converged': model._fit_result.converged if model._fit_result else False,
        'n_iterations': model._fit_result.n_iterations if model._fit_result else 0,
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(model_state, f)


def load_model(file_path: str) -> KLRfome:
    """
    Load a saved KLRfome model from disk.
    
    Parameters:
        file_path: Path to saved model
    
    Returns:
        KLRfome model instance (not fitted, but with saved hyperparameters)
    """
    with open(file_path, 'rb') as f:
        model_state = pickle.load(f)
    
    model = KLRfome(
        sigma=model_state['sigma'],
        lambda_reg=model_state['lambda_reg'],
        n_rff_features=model_state['n_rff_features'],
        window_size=model_state['window_size'],
        seed=model_state['seed']
    )
    
    # Note: Model needs to be refit or alpha needs to be restored
    # Full implementation would restore training data and alpha
    
    return model

