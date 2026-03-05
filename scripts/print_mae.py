import os, sys
# ensure project root is on sys.path so we can import raceAnalysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raceAnalysis import get_trained_model, DATA_DIR, CACHE_VERSION
model,mse,r2,mae,mean_err,evals_result,preprocessor = get_trained_model(20, CACHE_VERSION, os.path.getmtime(os.path.join(DATA_DIR,'f1ForAnalysis.csv')),'Position Group')
print('mae', mae)
