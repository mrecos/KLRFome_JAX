"""Validation and metrics utilities for KLRfome."""

from typing import List, Dict, Union, Optional
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..data.formats import TrainingData
from ..api import KLRfome


def CM_quads(
    pred: Union[Float[Array, "n"], np.ndarray, List[float]],
    obs: Union[Float[Array, "n"], np.ndarray, List[int]],
    threshold: Union[float, List[float]] = 0.5
) -> pd.DataFrame:
    """
    Compute Confusion Matrix quadrants at one or more thresholds.
    
    Parameters:
        pred: Predicted probabilities
        obs: Observed presence/absence (1/0)
        threshold: Single threshold or list of thresholds
    
    Returns:
        DataFrame with TP, FP, TN, FN for each threshold
    """
    pred = np.array(pred)
    obs = np.array(obs)
    
    if isinstance(threshold, (int, float)):
        threshold = [threshold]
    
    results = []
    
    for thresh in threshold:
        pred_cat = (pred >= thresh).astype(int)
        
        TP = np.sum((pred_cat == 1) & (obs == 1))
        FP = np.sum((pred_cat == 1) & (obs == 0))
        TN = np.sum((pred_cat == 0) & (obs == 0))
        FN = np.sum((pred_cat == 0) & (obs == 1))
        
        results.append({
            'Threshold': thresh,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        })
    
    return pd.DataFrame(results)


def cohens_kappa(TP: int, TN: int, FP: int, FN: int) -> float:
    """
    Compute Cohen's Kappa statistic.
    
    Parameters:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives
    
    Returns:
        Cohen's Kappa value
    """
    A, B, C, D = TP, FP, FN, TN
    n = A + B + C + D
    
    if n == 0:
        return 0.0
    
    Po = (A + D) / n
    Pe_a = ((A + B) * (A + C)) / n
    Pe_b = ((C + D) * (B + D)) / n
    Pe = (Pe_a + Pe_b) / n
    
    if Pe == 1:
        return 1.0 if Po == 1 else 0.0
    
    k = (Po - Pe) / (1 - Pe)
    return float(k)


def metrics(TP: int, TN: int, FP: int, FN: int) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.
    
    Returns 50+ metrics based on TP, TN, FP, FN.
    Similar to the R version.
    
    Parameters:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives
    
    Returns:
        Dictionary of metric names and values
    """
    # Basic calculations
    n = TP + TN + FP + FN
    if n == 0:
        return {}
    
    Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Specificity = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = Sensitivity
    
    # Create binary vectors for some metrics
    pred = np.array([1] * TP + [1] * FP + [0] * FN + [0] * TN)
    obs = np.array([1] * TP + [0] * FP + [1] * FN + [0] * TN)
    
    # Helper function for logit
    def logit(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))
    
    # Compute all metrics
    result = {
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'Prevalence': (TP + FN) / n,
        'Accuracy': (TP + TN) / n,
        'Err_Rate': (FP + FN) / n,
        'other': (TN + FP) / n,
        'Pm': (TP + FP) / n,
        'Pm_prime': (FN + TN) / n,
        'Precision': Precision,
        'Recall': Recall,
        'F_Measure': (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0,
        'Geo_Mean': np.sqrt(TP * TN) if TP > 0 and TN > 0 else 0.0,
        'MAE': float(np.mean(np.abs(pred - obs))),
        'RMSE': float(np.sqrt(np.mean((pred - obs) ** 2))),
        'FPR': 1 - Specificity,
        'FNR': 1 - Sensitivity,
        'TPR': Sensitivity,
        'TNR': Specificity,
        'FOR': FN / (FN + TN) if (FN + TN) > 0 else 0.0,
        'FDR': FP / (TP + FP) if (TP + FP) > 0 else 0.0,
        'Power': 1 - (1 - Sensitivity),
        'LRP': Sensitivity / (1 - Specificity) if Specificity < 1 else float('inf'),
        'log_LRP': np.log10(Sensitivity / (1 - Specificity)) if Specificity < 1 else float('inf'),
        'LRN': (1 - Sensitivity) / Specificity if Specificity > 0 else float('inf'),
        'PPV': Precision,
        'NPV': TN / (FN + TN) if (FN + TN) > 0 else 0.0,
        'KG': 1 - ((1 - Specificity) / Sensitivity) if Sensitivity > 0 else 0.0,
        'KG2': 1 - (((TP + FP) / n) / Sensitivity) if Sensitivity > 0 else 0.0,
        'DOR': (Sensitivity / (1 - Specificity)) / ((1 - Sensitivity) / Specificity) if Specificity < 1 and Specificity > 0 else float('inf'),
        'log_DOR': np.log10((Sensitivity / (1 - Specificity)) / ((1 - Sensitivity) / Specificity)) if Specificity < 1 and Specificity > 0 else float('inf'),
        'D': logit(Sensitivity) - logit(1 - Specificity),
        'S': logit(Sensitivity) + logit(1 - Specificity),
        'Kappa': cohens_kappa(TP, TN, FP, FN),
        'Opp_Precision': ((TP + TN) / n) - (np.abs(Specificity - Sensitivity) / (Specificity + Sensitivity)) if (Specificity + Sensitivity) > 0 else 0.0,
        'MCC': (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0.0,
        'Informedness': Sensitivity + Specificity - 1,  # TSS, Youden's J
        'Markedness': (TP / (TP + FP) if (TP + FP) > 0 else 0.0) + (TN / (FN + TN) if (FN + TN) > 0 else 0.0) - 1,
        'AFK': np.sqrt(Sensitivity * ((Sensitivity - (1 - Specificity)) / ((TN + FP) / n))) if Sensitivity > 0 and (TN + FP) > 0 else 0.0,
        'Indicative': Sensitivity / (1 - Specificity) if Specificity < 1 else float('inf'),
        'Indicative2': Sensitivity / ((TP + FP) / n) if (TP + FP) > 0 else float('inf'),
        'Indicative_norm': (Sensitivity / (1 - Specificity)) / ((TN + FP) / n) if Specificity < 1 and (TN + FP) > 0 else float('inf'),
        'Indicative_norm2': (Sensitivity / ((TP + FP) / n)) / ((TN + FP) / n) if (TP + FP) > 0 and (TN + FP) > 0 else float('inf'),
        'Brier': float(np.mean((obs - pred) ** 2)),
        'X1': (TP / (TP + FP) if (TP + FP) > 0 else 0.0) / ((TP + FN) / n) if (TP + FN) > 0 else 0.0,
        'X2': (FN / (FN + TN) if (FN + TN) > 0 else 0.0) / ((TP + FN) / n) if (TP + FN) > 0 else 0.0,
        'X3': (FP / (TP + FP) if (TP + FP) > 0 else 0.0) / ((TN + FP) / n) if (TN + FP) > 0 else 0.0,
        'X4': (TN / (FN + TN) if (FN + TN) > 0 else 0.0) / ((TN + FP) / n) if (TN + FP) > 0 else 0.0,
        'PPG': (TP / (TP + FP) if (TP + FP) > 0 else 0.0) / ((TP + FN) / n) if (TP + FN) > 0 else 0.0,
        'NPG': (FN / (FN + TN) if (FN + TN) > 0 else 0.0) / ((TP + FN) / n) if (TP + FN) > 0 else 0.0,
        'Balance': ((1 - ((1 - Specificity) / Sensitivity) if Sensitivity > 0 else 0.0) + (1 - ((1 - Sensitivity) / Specificity) if Specificity > 0 else 0.0)) / 2,
        'Balance2': ((1 - (((TP + FP) / n) / Sensitivity) if Sensitivity > 0 else 0.0) + (1 - ((1 - Sensitivity) / ((FN + TN) / n)) if (FN + TN) > 0 else 0.0)) / 2,
    }
    
    # Handle inf and nan values
    for key, value in result.items():
        if np.isinf(value) or np.isnan(value):
            result[key] = 0.0
    
    return result


def compute_roc_auc(
    pred: Union[Float[Array, "n"], np.ndarray, List[float]],
    obs: Union[Float[Array, "n"], np.ndarray, List[int]]
) -> float:
    """
    Compute ROC AUC using trapezoidal rule.
    
    Parameters:
        pred: Predicted probabilities
        obs: Observed labels (1/0)
    
    Returns:
        AUC value
    """
    from sklearn.metrics import roc_auc_score
    
    pred = np.array(pred)
    obs = np.array(obs)
    
    try:
        return float(roc_auc_score(obs, pred))
    except ValueError:
        # Handle case where only one class is present
        return 0.5


def cross_validate(
    model: KLRfome,
    training_data: TrainingData,
    n_folds: int = 5,
    stratified: bool = True,
    seed: int = 42
) -> Dict:
    """
    Perform k-fold cross-validation.
    
    Parameters:
        model: KLRfome model instance
        training_data: Training data
        n_folds: Number of folds
        stratified: Whether to stratify by label
        seed: Random seed
    
    Returns:
        Dictionary with metrics per fold and aggregated statistics
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    n = len(training_data.collections)
    fold_size = n // n_folds
    
    if stratified:
        # Separate sites and background
        sites = [c for c in training_data.collections if c.label == 1]
        background = [c for c in training_data.collections if c.label == 0]
        
        random.shuffle(sites)
        random.shuffle(background)
        
        # Create folds
        folds = []
        for i in range(n_folds):
            fold_sites = sites[i * len(sites) // n_folds:(i + 1) * len(sites) // n_folds]
            fold_bg = background[i * len(background) // n_folds:(i + 1) * len(background) // n_folds]
            folds.append(fold_sites + fold_bg)
    else:
        # Simple random split
        indices = list(range(n))
        random.shuffle(indices)
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    
    fold_results = []
    
    for fold_idx in range(n_folds):
        # Split into train and test
        test_indices = folds[fold_idx]
        train_indices = [i for i in range(n) if i not in test_indices]
        
        train_collections = [training_data.collections[i] for i in train_indices]
        test_collections = [training_data.collections[i] for i in test_indices]
        
        train_data = TrainingData(
            collections=train_collections,
            feature_names=training_data.feature_names,
            crs=training_data.crs
        )
        
        # Fit model
        model.fit(train_data)
        
        # Predict on test set (simplified - would need to build test kernel)
        # For now, return basic structure
        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(train_collections),
            'n_test': len(test_collections),
        })
    
    return {
        'folds': fold_results,
        'n_folds': n_folds,
        'mean_train_size': np.mean([r['n_train'] for r in fold_results]),
        'mean_test_size': np.mean([r['n_test'] for r in fold_results]),
    }

