from typing import List, Tuple, Any, Callable, Dict
import typing


class Function:
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List[type],
        return_type: type,
    ):
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.arity: int = len(self.arg_types)
        self.return_type: type = return_type

    def __str__(self):
        return self.name

    def vectorized_fn(self, in_values: Tuple[Tuple, ...]) -> Tuple:
        """
        Applies the function to a set of examples at once.
        The arg_vectors are nested tuples of shape (arity, num_examples).
        Returns an output "vector" of shape (num_examples).
        Not any more efficient currently, just cleans up the code a bit.
        Raises a SoftTypeError if the inputs are invalid types.
        """

        # gets the ith example from each node thats an input
        # so returns List of length arity
        def ith_examples(i: int) -> List:
            return [in_value[i] for in_value in in_values]

        num_examples = len(in_values[0])
        outputs = tuple(
            self.fn(*ith_examples(i))
            for i in range(num_examples))
        return outputs


class InverseFn:
    """
    An inverse function.
    """
    def __init__(self,
                 forward_fn: Callable,
                 inverse_fn: Callable[[Any], Tuple],
                 name: str = None):
        self.forward_fn = make_function(forward_fn)
        if name is None:
            self.name = self.forward_fn.name + '_inv'
        else:
            self.name = name
            assert 'inv' in name, "just would be nice to keep track of things"
        # maps output to tuple of inputs
        self.inverse_fn = inverse_fn

    def __str__(self):
        return self.name

    def vectorized_inverse(self, out_values: Tuple) -> Tuple[Tuple]:
        """
        Given the out values, produces a tuple of shape (num_inputs,
        num_examples) by calculating the inverse fn for each value.

        Raises a SoftTypeError if one of the outputs isn't valid to
        be inverted, or if the mask of None's isn't correct.
        """
        in_values = tuple(
            self.inverse_fn(out_value) for out_value in out_values)
        # go to tuple of shape (num_inputs, num_examples)
        in_values = tuple(zip(*in_values))
        return in_values


class CondInverseFn:
    """
    A conditional inverse function.
    """
    def __init__(self,
                 forward_fn: Callable,
                 inverse_fn: Callable[[Any, Tuple], Tuple],
                 name: str = None):
        self.forward_fn = make_function(forward_fn)
        if name is None:
            self.name = self.forward_fn.name + '_cond_inv'
        else:
            self.name = name
            assert 'cond_inv' in name, "just would be nice to keep track of things"
        # takes output and tuple of inputs, some of which are None masks.
        # e.g. for addition: self.inverse_fn(7, (3, None)) = (3, 4)
        self.inverse_fn = inverse_fn

    def __str__(self):
        return self.name

    def vectorized_inverse(self, out_values: Tuple,
                           in_values: Tuple[Tuple, ...]) -> Tuple[Tuple]:
        """
        Given input nodes (some of which are None) and the out value (a tuple
        of length num_examples) produces a tuple of shape (num_inputs,
        num_examples).

        Raises a SoftTypeError if one of the outputs isn't valid to
        be inverted, or if the mask of None's isn't correct.
        """

        # gets the ith example from each node thats an input
        # so returns List of length arity
        def ith_examples(i: int) -> List:
            return [None if in_value is None else in_value[i]
                    for in_value in in_values]

        num_examples = len(in_values[0])
        # tuple of shape (num_examples, num_inputs)
        full_in_values = [
            self.inverse_fn(out_values[i], ith_examples(i))
            for i in range(num_examples)
        ]

        # go to tuple of shape (num_inputs, num_examples)
        flipped_in_values = tuple(zip(*full_in_values))
        return flipped_in_values


def make_function(fn: Callable) -> Function:
    """
    Creates a Function for the given function. Infers types from type hints,
    so the op needs to be implemented with type hints.
    """
    types: Dict[str, type] = typing.get_type_hints(fn)
    if len(types) == 0:
        raise ValueError(("Operation provided does not use type hints, "
                          "which we use when choosing ops."))

    return Function(
        name=fn.__name__,
        fn=fn,
        # list of classes, one for each input arg. skip last type (return)
        arg_types=list(types.values())[0:-1],
        return_type=types["return"],
    )
