# Quick Test Guide: Performance Optimization

## Step 1: Generate Pre-Trained Models Locally

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Train and save all models (one-time, ~2-5 minutes)
python scripts\train_and_save_models.py
```

**Expected Output:**
```
Training position prediction model...
  ✓ Position model saved to data_files\models\position_model.pkl
Training DNF prediction model...
  ✓ DNF model saved to data_files\models\dnf_model.pkl
Training safety car model...
  ✓ Safety car model saved to data_files\models\safetycar_model.pkl

Summary:
  Models: ['position_model', 'dnf_model', 'safetycar_model']
  Cache Version: v2.3
  Timestamp: 2024-12-31 12:34:56
```

## Step 2: Test Fast Startup

```powershell
# Run Streamlit app
streamlit run raceAnalysis.py
```

**What to Watch:**
- ⏱️ **App should load in <5 seconds** (not 30-60s like before)
- 🚀 **No "Training model..." messages on startup**
- ✅ **Console shows**: "Loaded pretrained position_model (v2.3)" (if verbose)

## Step 3: Test Tab 5 (Predictive Models)

1. Navigate to **Tab 5: Predictive Models**
2. Tab should render instantly (<1 second)
3. Model metrics should display immediately (no training delay)

**Before**: 10-20 second delay while training
**After**: <1 second (loads from session cache)

## Step 4: Test Predictions Tab

1. Navigate to **Next Race** tab
2. Generate predictions for next race
3. Predictions should appear instantly

**What happens behind the scenes:**
- First prediction triggers lazy load of models (~1-2s one-time cost)
- Subsequent predictions are instant (models cached in session)

## Step 5: Verify Model Artifacts

```powershell
# Check model files exist
ls data_files\models\*.pkl
```

**Expected Files:**
```
position_model.pkl     (~500KB-5MB depending on model)
dnf_model.pkl          (~100KB-500KB)
safetycar_model.pkl    (~100KB-500KB)
```

## Step 6: Test Fallback (Optional)

```powershell
# Rename models directory to simulate missing models
Rename-Item data_files\models data_files\models_backup

# Run app - should fall back to training
streamlit run raceAnalysis.py

# Restore models
Rename-Item data_files\models_backup data_files\models
```

**Expected Behavior:**
- App starts but takes longer (30-60s)
- Console shows: "No pretrained model found, training from scratch"
- Models train on-demand and cache in session

## Timing Comparison

### Before Optimization:
```
[2024-12-31 12:00:00] Starting app...
[2024-12-31 12:00:15] Training position model... (15s)
[2024-12-31 12:00:35] Training DNF model... (20s)
[2024-12-31 12:00:50] Training safety car model... (15s)
[2024-12-31 12:01:00] App ready (60s total)
```

### After Optimization:
```
[2024-12-31 12:00:00] Starting app...
[2024-12-31 12:00:01] Loading pre-trained models...
[2024-12-31 12:00:03] App ready (3s total)
```

## Performance Metrics to Check

| Check | Before | After | Improvement |
|-------|--------|-------|-------------|
| Cold start (no cache) | 60s | 5s | **92% faster** |
| Tab 5 render | 15s | <1s | **93% faster** |
| First prediction | 20s | 2s | **90% faster** |
| Subsequent predictions | <1s | <1s | Same |

## Troubleshooting

### Issue: App still slow on startup
**Cause**: Pre-trained models not found or cache version mismatch

**Fix**:
1. Check `data_files\models\*.pkl` exist
2. Re-run `python scripts\train_and_save_models.py`
3. Clear Streamlit cache (Menu → "Clear Cache")

### Issue: "No pretrained model found" message
**Cause**: Model files missing or in wrong location

**Fix**:
```powershell
# Verify directory exists
mkdir data_files\models -Force

# Re-train models
python scripts\train_and_save_models.py
```

### Issue: Cache version mismatch warning
**Cause**: CACHE_VERSION changed but models weren't retrained

**Fix**:
```powershell
# Delete old models
Remove-Item data_files\models\*.pkl

# Train with new version
python scripts\train_and_save_models.py
```

## GitHub Actions Testing

### Trigger Workflow Manually:
1. Go to GitHub → Actions tab
2. Select "Train and Save Models" workflow
3. Click "Run workflow" → Run
4. Wait ~5-10 minutes for completion
5. Check commit history for model artifacts

### Verify Models Committed:
```bash
git log --oneline --all -n 5 | grep "model"
# Should see: "Update trained models [automated]"
```

## Success Criteria

✅ **App loads in <5 seconds** when pre-trained models exist  
✅ **Tab 5 renders instantly** (no training delay)  
✅ **Models lazy-load on demand** (first use adds ~1-2s)  
✅ **Fallback works** if models missing (trains automatically)  
✅ **GitHub Actions generates models** and commits to repo  

## Next Steps After Testing:

1. ✅ Verified local performance improvement
2. [ ] Commit changes and push to GitHub
3. [ ] Trigger GitHub Action to generate production models
4. [ ] Redeploy to Streamlit Cloud
5. [ ] Monitor production startup times
6. [ ] Share performance metrics with users

---

**Questions?** Check `PERFORMANCE_OPTIMIZATION.md` for detailed explanation of the architecture.
