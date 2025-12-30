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
    
    # Helper function to safely divide
    def safe_divide(numerator, denominator, default=0.0):
        """Safely divide, returning default if denominator is zero."""
        try:
            # Check for zero or very close to zero
            if denominator == 0 or (isinstance(denominator, (int, float)) and abs(denominator) < 1e-10):
                return default
            # Also check if it's a numpy array/scalar that's close to zero
            if hasattr(denominator, '__array__'):
                if np.abs(denominator) < 1e-10:
                    return default
            result = numerator / denominator
            # Check for inf or nan
            if np.isinf(result) or np.isnan(result):
                return default
            return result
        except (ZeroDivisionError, TypeError, ValueError):
            return default
    
    # Compute all metrics with safe division
    try:
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
        'LRP': safe_divide(Sensitivity, (1 - Specificity), default=float('inf')) if Specificity < 1 else float('inf'),
        'log_LRP': np.log10(safe_divide(Sensitivity, (1 - Specificity), default=float('inf'))) if Specificity < 1 and not np.isinf(safe_divide(Sensitivity, (1 - Specificity), default=float('inf'))) else float('inf'),
        'LRN': safe_divide((1 - Sensitivity), Specificity, default=float('inf')) if Specificity > 0 else float('inf'),
        'PPV': Precision,
        'NPV': TN / (FN + TN) if (FN + TN) > 0 else 0.0,
        'KG': 1 - ((1 - Specificity) / Sensitivity) if Sensitivity > 0 else 0.0,
        'KG2': 1 - (((TP + FP) / n) / Sensitivity) if Sensitivity > 0 else 0.0,
        'DOR': safe_divide(
            safe_divide(Sensitivity, (1 - Specificity), default=float('inf')),
            safe_divide((1 - Sensitivity), Specificity, default=float('inf')),
            default=float('inf')
        ) if Specificity < 1 and Specificity > 0 else float('inf'),
        'log_DOR': np.log10(safe_divide(
            safe_divide(Sensitivity, (1 - Specificity), default=float('inf')),
            safe_divide((1 - Sensitivity), Specificity, default=float('inf')),
            default=float('inf')
        )) if Specificity < 1 and Specificity > 0 and not np.isinf(safe_divide(
            safe_divide(Sensitivity, (1 - Specificity), default=float('inf')),
            safe_divide((1 - Sensitivity), Specificity, default=float('inf')),
            default=float('inf')
        )) else float('inf'),
        'D': logit(Sensitivity) - logit(1 - Specificity),
        'S': logit(Sensitivity) + logit(1 - Specificity),
        'Kappa': cohens_kappa(TP, TN, FP, FN),
        'Opp_Precision': ((TP + TN) / n) - (np.abs(Specificity - Sensitivity) / (Specificity + Sensitivity)) if (Specificity + Sensitivity) > 0 else 0.0,
        'MCC': (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0.0,
        'Informedness': Sensitivity + Specificity - 1,  # TSS, Youden's J
        'Markedness': (TP / (TP + FP) if (TP + FP) > 0 else 0.0) + (TN / (FN + TN) if (FN + TN) > 0 else 0.0) - 1,
        'AFK': np.sqrt(Sensitivity * safe_divide((Sensitivity - (1 - Specificity)), ((TN + FP) / n), default=0.0)) if Sensitivity > 0 and (TN + FP) > 0 and n > 0 else 0.0,
        'Indicative': safe_divide(Sensitivity, (1 - Specificity), default=float('inf')) if Specificity < 1 else float('inf'),
        'Indicative2': safe_divide(Sensitivity, ((TP + FP) / n), default=float('inf')) if (TP + FP) > 0 and n > 0 else float('inf'),
        'Indicative_norm': safe_divide(
            safe_divide(Sensitivity, (1 - Specificity), default=float('inf')),
            ((TN + FP) / n),
            default=float('inf')
        ) if Specificity < 1 and (TN + FP) > 0 and n > 0 else float('inf'),
        'Indicative_norm2': safe_divide(
            safe_divide(Sensitivity, ((TP + FP) / n), default=float('inf')),
            ((TN + FP) / n),
            default=float('inf')
        ) if (TP + FP) > 0 and (TN + FP) > 0 and n > 0 else float('inf'),
        'Brier': float(np.mean((obs - pred) ** 2)),
        'X1': safe_divide(safe_divide(TP, (TP + FP), default=0.0), ((TP + FN) / n), default=0.0) if (TP + FP) > 0 and (TP + FN) > 0 and n > 0 else 0.0,
        'X2': safe_divide(safe_divide(FN, (FN + TN), default=0.0), ((TP + FN) / n), default=0.0) if (FN + TN) > 0 and (TP + FN) > 0 and n > 0 else 0.0,
        'X3': safe_divide(safe_divide(FP, (TP + FP), default=0.0), ((TN + FP) / n), default=0.0) if (TP + FP) > 0 and (TN + FP) > 0 and n > 0 else 0.0,
        'X4': safe_divide(safe_divide(TN, (FN + TN), default=0.0), ((TN + FP) / n), default=0.0) if (FN + TN) > 0 and (TN + FP) > 0 and n > 0 else 0.0,
        'PPG': safe_divide(safe_divide(TP, (TP + FP), default=0.0), ((TP + FN) / n), default=0.0) if (TP + FP) > 0 and (TP + FN) > 0 and n > 0 else 0.0,
        'NPG': safe_divide(safe_divide(FN, (FN + TN), default=0.0), ((TP + FN) / n), default=0.0) if (FN + TN) > 0 and (TP + FN) > 0 and n > 0 else 0.0,
        'Balance': ((1 - ((1 - Specificity) / Sensitivity) if Sensitivity > 0 else 0.0) + (1 - ((1 - Sensitivity) / Specificity) if Specificity > 0 else 0.0)) / 2,
        'Balance2': ((1 - safe_divide(((TP + FP) / n), Sensitivity, default=0.0) if Sensitivity > 0 and n > 0 else 0.0) + (1 - safe_divide((1 - Sensitivity), ((FN + TN) / n), default=0.0) if (FN + TN) > 0 and n > 0 else 0.0)) / 2,
        }
    except (ZeroDivisionError, ValueError, TypeError) as e:
        # If any calculation fails, return empty dict or minimal metrics
        # This should rarely happen with all the guards, but just in case
        result = {
            'Sensitivity': Sensitivity,
            'Specificity': Specificity,
            'Precision': Precision,
            'Recall': Recall,
            'Accuracy': (TP + TN) / n if n > 0 else 0.0,
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
        if stratified:
            test_collections = folds[fold_idx]
            train_collections = []
            for i, coll in enumerate(training_data.collections):
                if coll not in test_collections:
                    train_collections.append(coll)
        else:
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
        
        # Build test kernel matrix: K_test shape (n_test, n_train)
        # Compute similarity between each test collection and each training collection
        distribution_kernel = model._distribution_kernel
        n_test = len(test_collections)
        n_train = len(train_collections)
        
        K_test = jnp.zeros((n_test, n_train))
        
        for i, test_coll in enumerate(test_collections):
            for j, train_coll in enumerate(train_collections):
                # Compute similarity between test and training collections
                similarity = distribution_kernel(test_coll.samples, train_coll.samples)
                K_test = K_test.at[i, j].set(similarity)
        
        # Get fitted alpha from model
        alpha = model._fit_result.alpha
        
        # Predict on test set
        klr = model._klr
        test_probs = klr.predict_proba(K_test, alpha)
        test_labels = jnp.array([coll.label for coll in test_collections])
        
        # Convert to numpy for metrics computation
        test_probs_np = np.array(test_probs)
        test_labels_np = np.array(test_labels)
        
        # Compute confusion matrix at threshold 0.5
        cm_df = CM_quads(test_probs_np, test_labels_np, threshold=0.5)
        if len(cm_df) > 0:
            TP = int(cm_df.iloc[0]['TP'])
            FP = int(cm_df.iloc[0]['FP'])
            TN = int(cm_df.iloc[0]['TN'])
            FN = int(cm_df.iloc[0]['FN'])
        else:
            TP = FP = TN = FN = 0
        
        # Compute metrics
        fold_metrics = metrics(TP, TN, FP, FN)
        
        # Compute AUC
        try:
            auc = compute_roc_auc(test_probs_np, test_labels_np)
        except Exception:
            auc = 0.5
        
        fold_metrics['AUC'] = auc
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(train_collections),
            'n_test': len(test_collections),
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'metrics': fold_metrics,
        })
    
    # Compute aggregated statistics
    metric_names = list(fold_results[0]['metrics'].keys())
    aggregated = {}
    
    for metric_name in metric_names:
        values = [r['metrics'][metric_name] for r in fold_results]
        aggregated[f'{metric_name}_mean'] = float(np.mean(values))
        aggregated[f'{metric_name}_std'] = float(np.std(values))
    
    # Find best fold (by AUC)
    best_fold_idx = max(range(len(fold_results)), 
                        key=lambda i: fold_results[i]['metrics'].get('AUC', 0.5))
    best_fold = fold_results[best_fold_idx]['fold']
    
    return {
        'folds': fold_results,
        'n_folds': n_folds,
        'mean_train_size': float(np.mean([r['n_train'] for r in fold_results])),
        'mean_test_size': float(np.mean([r['n_test'] for r in fold_results])),
        'aggregated_metrics': aggregated,
        'best_fold': best_fold,
    }

