import numpy as np

class CausalAssignmentModel:
    """
    Basically just a hack to allow me to provide information about the
    arguments of a dynamically generated function.
    """

    def __init__(self, model, parents):
        self.model = model
        self.parents = parents

    def __call__(self, *args, **kwargs):
        assert len(args) == 0
        return self.model(**kwargs)

    def __repr__(self):
        return "CausalAssignmentModel({})".format(",".join(self.parents))


# Some Helper functions for defining models

def discrete_model(parents, lookup_table):
    """
    Create CausalAssignmentModel based on a lookup table.

    Lookup_table maps inputs values to weigths of the output values
    The actual output values are sampled from a discrete distribution
    of integers with probability proportional to the weights.

    Lookup_table for the form:

    Dict[Tuple(input_vales): (output_weights)]

    Arguments
    ---------
    parents: list
        variable names of parents

    lookup_table: dict
        lookup table

    Returns
    -------
        model: CausalAssignmentModel
    """
    assert len(parents) > 0

    # create input/output mapping
    inputs, weights = zip(*lookup_table.items())

    output_length = len(weights[0])
    assert all(len(w) == output_length for w in weights)
    outputs = np.arange(output_length)
    ps = [np.array(w) / sum(w) for w in weights]  # probabilities

    def model(**kwargs):
        # n_samples = kwargs["n_samples"]
        print(kwargs)
        a = np.vstack([kwargs[p] for p in parents]).T

        b = [np.nan] * 1
        for m, p in zip(inputs, ps):
            b = np.where(
                (a == m).all(axis=1),
                np.random.choice(outputs, size=1, p=p), b)

        if np.isnan(b).any():
            raise ValueError(
                "It looks like an input was provided which doesn't have a lookup.")
        # print(b.astype(int))
        return b.astype(int)[0]

    return CausalAssignmentModel(model, parents)
