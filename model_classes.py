"""
Shared model class definitions for the F1 Analysis project.

Keeping these classes in a separate module is intentional: when ensemble pkl
files are deserialized by pickle.load, Python resolves class references by
importing the module recorded in each object's __module__ attribute.  If the
class lived inside raceAnalysis.py, deserialization would re-import the entire
Streamlit app (thousands of lines, including top-level st.* widget calls),
causing "duplicate widget key" and CachedWidgetWarning errors.  By defining
the class here, pickle only imports this lightweight module instead.
"""
from sklearn.base import BaseEstimator, RegressorMixin


class SklearnCompatibleCatBoost(BaseEstimator, RegressorMixin):
    """Wrapper for CatBoostRegressor to make it sklearn-compatible for ensemble stacking."""

    def __init__(self, **kwargs):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(**kwargs)
        self._estimator_type = "regressor"

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def __sklearn_tags__(self):
        """Manually implement sklearn tags to ensure proper regressor recognition."""
        from sklearn.utils._tags import Tags, TargetTags, RegressorTags, InputTags
        tags = Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            transformer_tags=None,
            regressor_tags=RegressorTags(),
            classifier_tags=None,
            array_api_support=False,
            no_validation=False,
            non_deterministic=False,
            requires_fit=True,
            _skip_test=False,
            input_tags=InputTags(pairwise=False),
        )
        return tags
