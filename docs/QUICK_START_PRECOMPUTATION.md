# Quick Start Guide: Precomputation System

## 🚀 Get Started in 5 Minutes

### Step 1: Verify Local Setup (Optional)

Test a precomputation script locally to ensure everything works:

```powershell
# Activate your venv (if not already active)
.\.venv\Scripts\Activate.ps1

# Test XGBoost model training (quick test, ~1-2 minutes)
python scripts/precompute/train_xgboost.py

# Check output
ls data_files/models/xgboost/
```

You should see JSON files created in `data_files/models/xgboost/`.

### Step 2: Push to GitHub

```powershell
# Stage all changes
git add .

# Commit
git commit -m "feat: Add comprehensive precomputation infrastructure

- Add 14 precomputation scripts for models and analyses
- Add 7 GitHub Actions workflows for automation
- Integrate precomputed data loaders into Streamlit UI
- Enable <3 second page loads via precomputation"

# Push to your repo
git push origin main
```

### Step 3: Enable GitHub Actions

1. Go to your GitHub repository
2. Click **Settings** tab
3. Scroll to **Actions** → **General**
4. Under "Actions permissions", select **Allow all actions and reusable workflows**
5. Click **Save**

### Step 4: Trigger Your First Workflow

#### Option A: Manual Trigger (Recommended for First Run)

1. Go to **Actions** tab in your repo
2. Click **Train All Models (Multi-Type)** workflow
3. Click **Run workflow** dropdown
4. Select branch: `main`
5. Click green **Run workflow** button
6. Watch the workflow run (~45 minutes)

#### Option B: Wait for Automatic Schedule

The master workflow runs **only during F1 season**:
- **Saturdays 12:00 AM UTC (Mar-Nov)** or **1:00 AM UTC (Dec-Feb)**
- Results committed by Sunday 8 AM UTC

**Seasonal Schedule Summary:**
- **Saturdays**: Model training (3 AM), position analysis (7 AM)
- **Sundays**: Feature selection (4-5 AM)
- **Fridays**: Race predictions (6 AM)
- **Quarterly**: Hyperparameter optimization (1st of Mar, Jun, Sep, Dec)

### Step 5: Pull Precomputed Results

After workflows complete:

```powershell
# Pull the auto-committed precomputed results
git pull origin main
```

### Step 6: View in Streamlit

```powershell
# Run the app
streamlit run raceAnalysis.py
```

Navigate to:
- **Tab 4 (🏁 Next Race)**: Choose "Use precomputed predictions (fast)"
- **Tab 5 (🤖 Predictive Models)**: Open "📦 View Precomputed Results"

## 📊 What to Expect

### First Workflow Run
- **Duration**: 45-120 minutes depending on workflow
- **Result**: JSON files in `data_files/precomputed/`
- **Automatic commit**: Workflows push results to repo with `[skip ci]` tag

### Subsequent Runs
- **Frequency**: Weekly (Sundays) automatically
- **Updates**: Fresh models, predictions, analyses
- **Incremental**: Only changed files are committed

## 🔍 Monitor Workflow Status

### GitHub Actions Tab
- **Green checkmark** ✅: Workflow succeeded
- **Red X** ❌: Workflow failed (click for logs)
- **Yellow circle** 🟡: Workflow running
- **Gray dash** ⚫: Workflow skipped/cancelled

### View Artifacts
1. Click on a completed workflow run
2. Scroll to **Artifacts** section
3. Download JSON results (available for 90 days)

### View Logs
1. Click on a workflow run
2. Click on a job (e.g., "train-xgboost")
3. Expand steps to see detailed logs
4. Look for errors if job failed

## 🛠️ Common Tasks

### Re-run a Failed Workflow
1. Go to the failed workflow run
2. Click **Re-run jobs** dropdown
3. Select **Re-run failed jobs**

### Trigger Predictions Before Race Weekend
```powershell
# Manually trigger predictions workflow
# (or just click in GitHub UI)
gh workflow run precompute-predictions.yml
```

### Update Precomputation Schedule
Edit workflow YAML files in `.github/workflows/`:

```yaml
schedule:
  - cron: '0 6 * * 4'  # Change time/day here
```

Use [crontab.guru](https://crontab.guru/) to generate cron expressions.

### Manual Workflow Execution Order
If you need to run workflows manually (during off-season or for testing), follow this dependency order:

1. **`train-all-models.yml`** ← Run first (no dependencies)
2. **Then run these in parallel:**
   - `feature-selection-suite.yml`
   - `feature-selection-monte-carlo.yml` 
   - `position-analysis.yml`
   - `precompute-predictions.yml`
3. **`hyperparameter-optimization.yml`** ← Run anytime (independent)

**Or simply run:** `weekly-precompute-all.yml` which orchestrates everything automatically.

### Email Notifications Setup
The system includes automated email notifications that send F1 predictions before race weekends:

**Workflow:** `send_predictions_schedule.yml`
- **Schedule:** Saturdays at 8:00 PM Eastern (seasonal, same as other workflows)
- **Purpose:** Automatically sends prediction emails using precomputed data
- **Dependencies:** Requires precomputed predictions from `precompute-predictions.yml`

**Required GitHub Secrets:**
```
EMAIL_FROM      # Sender email address
EMAIL_TO         # Recipient email address  
EMAIL_PASSWORD   # Email password or app password
SMTP_SERVER      # SMTP server (e.g., smtp.gmail.com)
SMTP_PORT        # SMTP port (e.g., 587 for TLS)
```

**Setup Steps:**
1. Go to repository Settings → Secrets and variables → Actions
2. Add the 5 required secrets above
3. The workflow will automatically send emails during F1 season
4. Use manual trigger for testing during off-season

**Email Content:**
- Race predictions with MAE metrics
- Driver and constructor rankings
- Precomputed analysis results
- Links to full Streamlit dashboard

### Disable a Workflow
Add to the top of the workflow YAML:

```yaml
on:
  workflow_dispatch:  # Only manual triggers
  # schedule:  # Commented out to disable automatic runs
  #   - cron: '0 6 * * 4'
```

## 📁 File Locations

### Precomputed Results
```
data_files/precomputed/
├── predictions/
│   └── predictions_Bahrain_Grand_Prix_2025.json
├── monte_carlo_results.json      ← MAE-optimal feature subsets
├── shap_results.json              ← Feature importance rankings
├── rfe_results.json               ← RFE selected features
├── boruta_results.json            ← Boruta all-relevant features
├── permutation_results.json       ← Permutation importance
├── hyperparam_bayesian.json       ← Optuna optimization results
├── hyperparam_grid.json           ← Grid search results
└── position_mae_detailed.json     ← Position-specific MAE
```

### Trained Models
```
data_files/models/
├── xgboost/
│   ├── position_model.json        ← Final position predictions
│   ├── dnf_model.json            ← DNF probability
│   └── safety_car_model.json     ← Safety car likelihood
├── lightgbm/
│   └── position_model.txt
├── catboost/
│   └── position_model.cbm
└── ensemble/
    └── position_model.pkl
```

## 🎯 Best Practices

### Development Workflow
1. **Test locally first**: Run scripts with `--help` to see options
2. **Commit incrementally**: Push changes as you test them
3. **Monitor Actions**: Watch first few runs to ensure success
4. **Pull regularly**: `git pull` to get latest precomputed results
5. **Seasonal awareness**: Workflows only run during F1 season (Mar-Dec)

### Production Workflow
1. **Let automation run**: Don't manually trigger unless needed during season
2. **Check Saturdays**: Verify weekly runs completed successfully during F1 season
3. **Before races**: Predictions auto-update Fridays during season
4. **Quarterly review**: Check hyperparameter optimization results (Mar, Jun, Sep, Dec)
5. **Off-season**: Use manual triggers if testing needed

### Debugging
1. **Check workflow logs** for error messages
2. **Verify data files exist** in `data_files/`
3. **Test scripts locally** before pushing changes
4. **Use `--help`** flag on all scripts to see options

## 📈 Performance Monitoring

### Track Page Load Times
Before:
```
Tab 4 load: 30-60 seconds
Feature selection: 2-15 minutes
```

After:
```
Tab 4 load: <3 seconds ✅
Feature selection: <1 second ✅
```

### Track MAE Improvements
Monitor `position_mae_detailed.json` weekly during F1 season:
- Current: ~1.94
- Target: ≤1.5
- Position-specific MAE for focused optimization

## 📈 Performance Monitoring

### Track Page Load Times
Before:
```
Tab 4 load: 30-60 seconds
Feature selection: 2-15 minutes
```

After:
```
Tab 4 load: <3 seconds ✅
Feature selection: <1 second ✅
```

### Track MAE Improvements
Monitor `position_mae_detailed.json` weekly during F1 season:
- Current: ~1.94
- Target: ≤1.5
- Position-specific MAE for focused optimization

## 💡 Tips & Tricks

### Speed Up Development
```powershell
# Use smaller trial counts for testing
python scripts/precompute/monte_carlo_features.py --n-trials 10

# Test single model instead of ensemble
python scripts/precompute/train_xgboost.py
```

### View JSON Results
```powershell
# Pretty-print JSON
python -m json.tool data_files/precomputed/monte_carlo_results.json

# Or use VSCode JSON viewer
code data_files/precomputed/monte_carlo_results.json
```

### Workflow Notifications
Add to workflow YAML for Slack/Discord notifications:

```yaml
- name: Notify on completion
  if: always()
  run: |
    # Send notification here
    # (Slack webhook, Discord webhook, email, etc.)
```

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Workflows complete with green checkmarks **during F1 season**
- ✅ JSON files appear in `data_files/precomputed/` after runs
- ✅ Streamlit UI shows "Precomputed results available"
- ✅ Tab 4 loads in <3 seconds
- ✅ Predictions update weekly during season automatically
- ✅ No workflow runs during off-season (Dec-Feb gap)

## 🆘 Get Help

### Common Issues

**Q: Workflow fails with "No module named 'streamlit'"**  
A: Add dependency to `requirements.txt` and push

**Q: Predictions not loading in UI**  
A: Check `data_files/precomputed/predictions/` exists and has JSON files

**Q: Models not updating**  
A: Verify workflows completed successfully and `git pull` latest changes

**Q: JSON parse errors**  
A: Check workflow logs for script errors, fix and re-run

**Q: Workflows not running automatically**  
A: Check if it's F1 season (Mar-Dec). Use manual trigger during off-season for testing

**Q: No workflow runs during off-season (Dec-Feb)**  
A: This is expected! Workflows only run during F1 season to save resources. Use manual triggers for testing.

**Q: GitHub Actions quota exceeded**  
A: You're using >2,000 min/month. Workflows only run during F1 season to minimize usage.

### Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Workflow YAML Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

## 🚀 You're Ready!

Follow the 6 steps above and you'll have a fully automated precomputation system running within the hour. The Streamlit app will load dramatically faster and provide fresh predictions weekly without any manual intervention.

**Seasonal Operation Note:** All workflows are configured to run only during F1 season (March-December) to optimize resource usage. During off-season (December-February), workflows can be triggered manually for testing but won't run automatically.

**Happy racing! 🏎️💨**
