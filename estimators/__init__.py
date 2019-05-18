from abc import ABC, abstractmethod


class Estimator(ABC):
    def fit(self, x, y=None, **fit_params):
        return self

    @abstractmethod
    def predict(self, x):
        print ("predicting" * 100)
        raise NotImplementedError

    def transform(self, X):
        """Implements identity transform: the output equals to the input.

        Motivation: all members in the pipeline, up to this estimator, are expected to
        support transform operation. Consequently, if we add identity transform in here,  we are able to run:
        pipeline.transform(pipeline_input) and smoothly retrieve the transformed input, which is fed into
        the estimator on running 'predict'.
        """

        return X
