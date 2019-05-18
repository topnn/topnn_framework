import glob
from inspect import signature
from os.path import dirname, basename, isfile
from typing import Any, Union, GenericMeta

from sklearn.pipeline import Pipeline

modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


class InvalidPipelineException(TypeError):
    pass


def validate(pipeline: Pipeline):
    """
    This is a very basic Pipeline validator. It checks whether the input/output types of steps are correct.

    At the moment it only supports simple types and Union. A full validator is much more complicated.
    There's a tool that can potentially be used for this purpose: http://www.mypy-lang.org
    """
    output_type = None
    for step_name, step in pipeline.steps:
        try:
            sig = signature(step.transform)
        except AttributeError:
            sig = signature(step.predict)

        input_type = next(iter(sig.parameters.values())).annotation

        if output_type is None:  # first element of pipeline
            output_type = input_type

        allowed_input_types = input_type.__args__ if type(input_type) == type(Union) else [input_type]
        allowed_output_types = output_type.__args__ if type(output_type) == type(Union) else [output_type]

        any_valid = False
        for x in allowed_output_types:
            if x is Any:
                any_valid = True
                break

            if x in allowed_input_types:
                any_valid = True
                break

            for parent in allowed_input_types:
                if type(parent) is GenericMeta:
                    # TODO I couldn't find a better way of converting typing.List to list
                    parent = list

                if issubclass(x, parent) or issubclass(parent, x):
                    any_valid = True
                    break

        if not any_valid:
            raise InvalidPipelineException(
                "Input type of `{}` step does not match output of previous step".format(step_name))

        output_type = sig.return_annotation

    return True
