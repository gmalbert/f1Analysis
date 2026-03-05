"""Smoke test for ROADMAP-3 components."""
import re, sys
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import TargetEncoder, RobustScaler, StandardScaler
import numpy as np
import pandas as pd

# 1. IterativeImputer
X = pd.DataFrame({'a': [1.0, None, 3.0, 4.0], 'b': [2.0, 3.0, None, 5.0]})
out = IterativeImputer(max_iter=5, random_state=42, skip_complete=True).fit_transform(X)
assert not np.isnan(out).any(), 'NaNs remain after IterativeImputer'
print('IterativeImputer: OK')

# 2. TargetEncoder
X2 = pd.DataFrame({'team': ['red', 'blue', 'red', 'blue', 'green']})
y2 = [1.0, 2.0, 1.5, 2.5, 3.0]
out2 = TargetEncoder(target_type='continuous', random_state=42).fit_transform(X2, y2)
assert out2.shape == (5, 1)
print('TargetEncoder: OK')

# 3. raceAnalysis.py - class & function definitions present
src = open('raceAnalysis.py', encoding='utf-8').read()
for name in ['PositionGroupEnsemble', 'TrackWeightedEnsemble',
             'CIRCUIT_ENSEMBLE_WEIGHTS', '_build_advanced_preprocessor']:
    assert name in src, f'Missing in raceAnalysis.py: {name}'
# Position Group training is inlined inside train_and_evaluate_model
assert 'Position Group' in src and 'PositionGroupEnsemble(' in src, 'Position Group training branch missing'
print('All class/function definitions in raceAnalysis.py: OK')

# 4. CACHE_VERSION
m = re.search(r'CACHE_VERSION\s*=\s*"([^"]+)"', src)
assert m
print(f'CACHE_VERSION = {m.group(1)}')

# 5. Position Group in model selector list
assert '"Position Group"' in src or "'Position Group'" in src
print('Position Group model option present: OK')

# 6. monthly_hpo.py exists and compiles
import py_compile
py_compile.compile('scripts/precompute/monthly_hpo.py', doraise=True)
print('monthly_hpo.py compiles: OK')

# 7. monthly-hpo.yml workflow exists
from pathlib import Path
assert Path('.github/workflows/monthly-hpo.yml').exists()
print('monthly-hpo.yml workflow exists: OK')

print('\nAll smoke checks passed.')
