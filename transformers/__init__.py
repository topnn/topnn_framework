from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin, BaseEstimator


class Transformer(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def transform(self, x):
        raise NotImplementedError

    def fit(self, x, y=None, **fit_params):
        return self