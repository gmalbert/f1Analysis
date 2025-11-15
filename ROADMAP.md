# F1 Analysis App Roadmap

[← Back to README](README.md)

## Project Mission
Minimize MAE (Mean Absolute Error) for Formula 1 race predictions with a target of **MAE ≤ 1.5** for final position predictions.

---

## Current State
- ✅ 6-tab Streamlit UI (Data Explorer, Analytics, Current Season, Next Race, Predictive Models, Raw Data)
- ✅ XGBoost model with 70+ engineered features
- ✅ Position group analysis (Winners, Podium, Points, Mid-field, Back-field)
- ✅ Multi-source data integration (F1DB, FastF1, Open-Meteo)
- ✅ Monte Carlo simulation for feature validation (1000 iterations)
- ✅ Current MAE: ~1.94 (target: ≤1.5)

---

## Phase 1: Model Performance Optimization (Priority: HIGH)
**Goal:** Reduce MAE below 1.5 consistently across all position groups

### 1.1 Feature Engineering Enhancements
- [x] **Driver form momentum features**
  - Rolling 3-race win percentage
  - Recent qualifying improvement trend
  - Head-to-head teammate performance delta
  - Championship position pressure factor
  
- [x] **Constructor reliability features**
  - Recent mechanical DNF rate by constructor
  - Engine penalty impact on grid position
  - Component age vs. failure probability
  - Constructor development rate (mid-season upgrades)

- [x] **Track-specific intelligence**
  - Driver historical performance at specific circuit types (street, high-speed, technical)
  - Weather pattern analysis by circuit location
  - Tire compound strategy effectiveness by track
  - Overtaking difficulty index per circuit

- [x] **Qualifying-to-race correlation**
  - Q1/Q2/Q3 sector time consistency
  - Qualifying position vs. race pace delta by track
  - Tire compound used in qualifying vs. race start
  - Traffic impact on qualifying laps

### 1.2 Model Architecture Experiments
- [x] **Ensemble methods**
  - Stack XGBoost + LightGBM + CatBoost
  - Weighted average based on track type
  - Position-specific models (separate models for P1-3, P4-10, P11-20)

- [x] **Hyperparameter optimization**
  - Bayesian optimization for XGBoost parameters
  - Grid search for learning rate, max_depth, min_child_weight
  - Cross-validation strategy improvements (stratified by season)

- [ ] **Feature selection refinement**
  - Re-run Boruta feature selection with new features
  - SHAP value analysis for feature importance
  - Remove redundant/highly correlated features
  - Test feature interactions (polynomial features for top predictors)

### 1.3 Data Quality & Leakage Prevention
- [ ] **Audit current features for temporal leakage**
  - Verify no future-looking data in training set
  - Check practice/qualifying data availability timing
  - Ensure safety car features don't include race outcome data

- [ ] **Missing data handling**
  - Imputation strategy for missing practice sessions
  - Sprint race weekend data handling
  - Weather data gaps filling methodology

---

## Phase 2: Prediction Capabilities Expansion (Priority: MEDIUM)

### 2.1 New Prediction Types
- [ ] **DNF probability predictions**
  - Binary classification for finish/DNF
  - Multi-class classification for DNF reason (mechanical, accident, disqualification)
  - Lap-by-lap DNF risk scoring

- [ ] **Pit stop timing predictions**
  - Optimal pit window prediction
  - Number of stops prediction by strategy
  - Undercut/overcut success probability

- [ ] **Race winner probability**
  - Pre-race winner probability by driver
  - Dynamic in-race probability updates (if live data available)
  - Championship points impact scenarios

### 2.2 Live Race Integration (Future)
- [ ] **Real-time prediction updates**
  - FastF1 live timing integration
  - Dynamic MAE calculation as race progresses
  - Strategy change impact on predictions

- [ ] **What-if scenario modeling**
  - Safety car deployment impact
  - Weather change scenarios
  - Virtual safety car vs. full safety car effects

---

## Phase 3: UI/UX Enhancements (Priority: MEDIUM)

### 3.1 Interactive Visualizations
- [ ] **Enhanced position group analysis**
  - Add confidence intervals to predictions
  - Show historical MAE trends over seasons
  - Driver/constructor heatmaps for track performance

- [ ] **Prediction explainability**
  - SHAP force plots for individual race predictions
  - Feature contribution breakdown per driver
  - "Why did X finish Yth?" explanations

- [ ] **Comparative analysis tools**
  - Driver vs. driver prediction comparison
  - Team vs. team performance trends
  - Season-over-season improvement tracking

### 3.2 User Experience
- [ ] **Performance optimizations**
  - Cache frequently accessed data
  - Lazy load heavy visualizations
  - Pagination for large datasets

- [ ] **Export capabilities**
  - Download predictions as CSV
  - Export charts as PNG/SVG
  - Generate PDF race reports

- [ ] **Mobile responsiveness**
  - Optimize layout for smaller screens
  - Touch-friendly controls
  - Simplified mobile views

---

## Phase 4: Data Infrastructure (Priority: LOW)

### 4.1 Data Pipeline Automation
- [ ] **Scheduled data updates**
  - Automatic F1DB JSON refresh
  - FastF1 cache maintenance
  - Weather data backfill for missed races

- [ ] **Data validation**
  - Automated consistency checks
  - Missing data alerts
  - Outlier detection and flagging

### 4.2 Database Integration (Optional)
- [ ] **Move from CSV to database**
  - PostgreSQL or SQLite for better performance
  - Query optimization for large datasets
  - Version control for data snapshots

---

## Phase 5: Advanced Analytics (Priority: LOW)

### 5.1 Championship Modeling
- [ ] **Season-long predictions**
  - Driver championship winner probability
  - Constructor championship projections
  - Points distribution scenarios

### 5.2 Strategy Analysis
- [ ] **Tire strategy optimization**
  - Optimal tire compound selection
  - Stint length predictions
  - Degradation modeling by compound/track

- [ ] **Qualifying strategy**
  - Q1/Q2/Q3 progression probability
  - Tire saving vs. track position tradeoffs

### 5.3 Historical Insights
- [ ] **Career trajectory modeling**
  - Driver peak performance prediction
  - Rookie vs. veteran performance curves
  - Age vs. performance correlation

---

## Success Metrics

### Primary KPIs
- **MAE < 1.5** for final position predictions (maintain below threshold)
- **MAE < 0.8** for podium positions (P1-3)
- **MAE < 1.2** for points positions (P1-10)
- **DNF prediction accuracy > 75%** (when implemented)
- **Pit stop timing MAE < 2 laps** (when implemented)

### Secondary Metrics
- User engagement (session duration, feature usage)
- Prediction confidence calibration (predicted probability vs. actual outcomes)
- Feature engineering velocity (new features added per sprint)
- Model retraining frequency and stability

---

## Technical Debt & Maintenance

### Code Quality
- [ ] Add comprehensive unit tests for feature engineering functions
- [ ] Add integration tests for data pipeline
- [ ] Document all feature calculations with formulas
- [ ] Add type hints throughout codebase
- [ ] Refactor `f1-generate-analysis.py` (2500+ lines) into modules

### Documentation
- [ ] Create feature engineering cookbook
- [ ] Document model training process
- [ ] Add API documentation for reusable functions
- [ ] Create troubleshooting guide

### Monitoring
- [ ] Track MAE by season, track, position group
- [ ] Monitor data freshness and completeness
- [ ] Alert on prediction anomalies
- [ ] Log model performance degradation

---

## Quick Wins (Start Here)

These items can deliver immediate MAE improvements with minimal effort:

1. **Add driver momentum features** (recent win rate, qualifying trends)
2. **Implement position-specific models** (separate model for P1-3 vs P4-10 vs P11-20)
3. **Add confidence intervals** to predictions in UI
4. **Cache expensive computations** in Streamlit app
5. **Add track type classification** (street, high-speed, technical) as categorical feature
6. **Implement hyperparameter tuning** with cross-validation
7. **Add feature interaction terms** for top 10 features by SHAP value

---

## Long-term Vision

**Ultimate Goal:** Create the most accurate F1 race prediction system by:
- Achieving MAE < 1.0 for all position predictions
- Providing real-time race strategy recommendations
- Offering comprehensive "what-if" scenario modeling
- Building a prediction API for external consumption
- Publishing MAE benchmarks and methodology as open research

---

## Resource Requirements

### Immediate Needs
- Historical tire compound data (not fully integrated)
- Live timing API access (for real-time predictions)
- Additional computing resources for ensemble models

### Future Considerations
- Cloud hosting for production deployment
- Database infrastructure for scaling
- CDN for static assets (charts, cached data)

---

## Review Schedule
- **Weekly:** MAE tracking and feature experiment results
- **Bi-weekly:** Roadmap prioritization updates
- **Monthly:** Model retraining and performance audit
- **Quarterly:** Strategic direction and vision alignment

---

**Last Updated:** November 14, 2025  
**Next Review:** November 21, 2025
