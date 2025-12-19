import sys
sys.path.insert(0, r'c:/Users/gmalb/Downloads/f1Analysis')
import importlib, raceAnalysis
importlib.reload(raceAnalysis)
mod = getattr(raceAnalysis, 'audit_temporal_leakage', None)
print('audit_temporal_leakage is', type(mod))
print('module file:', getattr(mod, '__file__', None))
print('module repr:', repr(mod))
