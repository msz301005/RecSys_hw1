"""Rating- and recency-aware target, holding confidence equal to base iALS."""
import numpy as np
from .ials import ImplicitALS


class CustomTargetALS(ImplicitALS):
    def __init__(self, half_life_days=365.0, freshness_floor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days, self.freshness_floor = half_life_days, freshness_floor

    def make_target(self, ratings, timestamps, cutoff):
        if self.half_life_days <= 0 or not 0 <= self.freshness_floor <= 1:
            raise ValueError("Invalid recency parameters")
        if np.any(timestamps.data >= cutoff):
            raise ValueError("Training contains events from or after the recommendation time")
        age_days = (cutoff - timestamps.data) / 86400.0
        freshness = self.freshness_floor + (1 - self.freshness_floor) * np.exp(-np.log(2) * age_days / self.half_life_days)
        result = ratings.astype(np.float64).copy()
        result.data = ((result.data - 3.0) / 2.0) * freshness
        return result

    def fit(self, observed, ratings, timestamps, cutoff):
        self.target = self.make_target(ratings, timestamps, cutoff)
        return super().fit(observed, self.target)
