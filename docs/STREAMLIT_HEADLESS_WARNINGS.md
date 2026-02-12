# Streamlit Headless Mode Warnings

## Summary
All 13 precompute scripts use `STREAMLIT_LOG_LEVEL='error'` to minimize Streamlit logging when running in headless mode (GitHub Actions or local execution without the Streamlit server).

## Expected Behavior
When running precompute scripts, you'll see informational warnings like:
```
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
No runtime found, using MemoryCacheStorageManager
```

These warnings:
- ✅ Are **informational only** - they explicitly state "can be ignored"
- ✅ Do **not** affect script functionality (models train successfully)
- ✅ Are **expected** when running Streamlit code outside the Streamlit server
- ✅ Cannot be completely suppressed at the Python logging level (emitted during import)

## Verification
To verify scripts work correctly, filter for actionable output:

**PowerShell:**
```powershell
python scripts\precompute\train_ensemble.py 2>&1 | Select-String -Pattern "\[OK\]|\[ERROR\]|MAE|Training|===="
```

**Expected clean output:**
```
============================================================
Ensemble Model Training
============================================================
Loading data (CACHE_VERSION=v2.5)...
Training Ensemble Model (XGBoost + LightGBM + CatBoost)
[OK] Ensemble model saved (MAE: 2.0686)
============================================================
Ensemble Training Complete!
============================================================
Position MAE: 2.0686
```

## GitHub Actions
If you want to suppress these warnings in GitHub Actions logs, add to your workflow YAML:

```yaml
- name: Train Ensemble Model
  run: |
    python scripts/precompute/train_ensemble.py 2>&1 | grep -E "\[OK\]|\[ERROR\]|MAE|Training|===="
```

## Configuration Applied
All 13 precompute scripts have:

```python
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'  # Minimize Streamlit logging

import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
```

This represents best-practice configuration - further suppression would require not using Streamlit components in headless scripts (major refactoring).

## Scripts Configured
1. `train_xgboost.py`
2. `train_lightgbm.py`
3. `train_catboost.py`
4. `train_ensemble.py`
5. `monte_carlo_features.py`
6. `generate_race_predictions.py`
7. `boruta_features.py`
8. `rfe_features.py`
9. `shap_analysis.py`
10. `hyperparameter_bayesian.py`
11. `hyperparameter_grid_search.py`
12. `permutation_importance.py`
13. `position_group_analysis_precompute.py`

## Alternatives Attempted
We attempted multiple suppression approaches:
1. ❌ `warnings.filterwarnings("ignore")` - doesn't affect Streamlit internals
2. ❌ Python logging configuration before import - warnings emitted during import
3. ❌ Python logging configuration after import - warnings already emitted
4. ❌ `sys.stderr` redirection - warnings persist throughout execution
5. ✅ `STREAMLIT_LOG_LEVEL='error'` - **best practice** (minimizes but doesn't eliminate)

The warnings stem from Streamlit's architecture detecting headless execution and are informational by design.
