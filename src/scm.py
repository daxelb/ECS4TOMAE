# Besides some added methods, most of this class was written by Iain Barr (ijmbarr on GitHub)
# from his public repository, causalgraphicalmodels
# The code has been imported and modified into this project for ease/consistency

import numpy as np
from causalgraphicalmodels.csm import StructuralCausalModel, linear_model, logistic_model
import inspect
import numpy as np
import pandas as pd
import networkx as nx
import os

from cgm import CausalGraph

class SCM:
    def __init__(self, assignment):
        """
        Creates StructuralCausalModel from assignment of the form
        { variable: Function(parents) }
        """

        self.assignment = assignment.copy()
        nodes = list(assignment.keys())
        set_nodes = []
        edges = []

        for node, model in assignment.items():
            if model is None:
                set_nodes.append(node)

            elif isinstance(model, CausalAssignmentModel):
                edges.extend([
                    (parent, node)
                    for parent in model.parents
                ])

            elif callable(model):
                sig = inspect.signature(model)
                parents = [
                    parent
                    for parent in sig.parameters.keys()
                    if parent != "n_samples"
                ]
                self.assignment[node] = CausalAssignmentModel(model, parents)
                edges.extend([(p, node) for p in parents])

            else:
                raise ValueError("Model must be either callable or None. "
                                 "Instead got {} for node {}."
                                 .format(model, node))

        self.cgm = CausalGraph(
            nodes=nodes, edges=edges, set_nodes=set_nodes)

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.cgm.dag.nodes())))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))
              
    def get_variables(self):
        return sorted(list(self.cgm.dag.nodes))

    def get_parents(self, node):
        return sorted(list(self.cgm.dag.predecessors(node)))

    def get_children(self, node):
        return sorted(list(self.cgm.dag.successors(node)))

    def get_exogenous(self):
        return sorted([n for n in self.cgm.dag.nodes if not self.get_parents(n)])

    def get_endogenous(self):
        return sorted([n for n in self.cgm.dag.nodes if self.get_parents(n)])

    def get_leaves(self):
        leaves = list()
        for i in self.cgm.dag.nodes:
            if not len(list(self.cgm.dag.successors(i))):
                leaves.append(i)
        return leaves

    def draw_model(self, v=False):
        self.cgm.draw().render('output/SCM.gv', view=v)

    def get_distribution(self):
        return self.cgm.get_distribution()

    def is_d_separated(self, x, y, zs=None):
        return self.cgm.is_d_separated(x, y, zs)

    def get_all_independence_relationships(self):
        return self.cgm.get_all_independence_relationships()

    def get_all_backdoor_paths(self, x, y):
        return self.cgm.get_all_backdoor_paths(x, y)

    def is_valid_backdoor_adjustment_set(self, x, y, z):
        return self.cgm.is_valid_backdoor_adjustment_set(x, y, z)

    def get_all_backdoor_adjustment_sets(self, x, y):
        return self.cgm.get_all_backdoor_adjustment_sets(x, y)

    def is_valid_frontdoor_adjustment_set(self, x, y, z):
        return self.cgm.is_valid_frontdoor_adjustment_set(x, y, z)

    def get_all_frontdoor_adjustment_sets(self):
        return self.cgm.get_all_frontdoor_adjustment_sets()

    def sample(self, n_samples=100, set_values=None):
        """
        Sample from CSM

        Arguments
        ---------
        n_samples: int
            the number of samples to return

        set_values: dict[variable:str, set_value:np.array]
            the values of the interventional variable

        Returns
        -------
        samples: pd.DataFrame
        """
        samples = {}

        if set_values is None:
            set_values = dict()

        for node in nx.topological_sort(self.cgm.dag):
            c_model = self.assignment[node]

            if c_model is None:
                assert len(set_values[node]) == n_samples
                samples[node] = np.array(set_values[node])
            else:
                parent_samples = {
                    parent: samples[parent]
                    for parent in c_model.parents
                }
                parent_samples["n_samples"] = n_samples
                samples[node] = c_model(**parent_samples)

        return pd.DataFrame(samples)

    def do(self, node):
        """
        Returns a StructualCausalModel after an intervention on node
        """
        new_assignment = self.assignment.copy()
        new_assignment[node] = None
        return StructuralCausalModel(new_assignment)


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

def _sigma(x):
    return 1 / (1 + np.exp(-x))


def linear_model(parents, weights, offset=0, noise_scale=1):
    """
    Create CausalAssignmentModel for node y of the form
    \sum_{i} x_{i}w_{i} + a + \epsilon

    Arguments
    ---------
    parents: list
        variable names of parents

    weights: list
        weigths of each variable in the sum

    offset: float
        offset for sum

    noise_scale: float
        scale of the normal noise

    Returns
    -------
        model: CausalAssignmentModel
    """
    assert len(parents) == len(weights)
    assert len(parents) > 0

    def model(**kwargs):
        n_samples = kwargs["n_samples"]
        a = np.array(
            [kwargs[p] * w for p, w in zip(parents, weights)], dtype=np.float)
        a = np.sum(a, axis=0)
        a += np.random.normal(loc=offset, scale=noise_scale, size=n_samples)
        return a

    return CausalAssignmentModel(model, parents)


def logistic_model(parents, weights, offset=0):
    """
    Create CausalAssignmentModel for node y of the form
    z = \sum_{i} x_{i}w_{i} + a
    y ~ Binomial(\simga(z))

    Arguments
    ---------
    parents: list
        variable names of parents

    weights: list
        weigths of each variable in the sum

    offset: float
        offset for sum

    Returns
    -------
        model: CausalAssignmentModel
    """
    assert len(parents) == len(weights)
    assert len(parents) > 0

    def model(**kwargs):
        a = np.array([kwargs[p] * w for p, w in zip(parents, weights)])
        a = np.sum(a, axis=0) + offset
        a = _sigma(a)
        a = np.random.binomial(n=1, p=a)
        return a

    return CausalAssignmentModel(model, parents)


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

    ps = [np.array(w) / sum(w) for w in weights]

    def model(**kwargs):
        n_samples = kwargs["n_samples"]
        a = np.vstack([kwargs[p] for p in parents]).T

        b = np.zeros(n_samples) * np.nan
        for m, p in zip(inputs, ps):
            b = np.where(
                (a == m).all(axis=1),
                np.random.choice(outputs, size=n_samples, p=p), b)

        if np.isnan(b).any():
            raise ValueError(
                "It looks like an input was provided which doesn't have a lookup.")

        return b

    return CausalAssignmentModel(model, parents)

if __name__ == "__main__":
  os.environ["PATH"] += os.pathsep + \
      'C:/Program Files/Graphviz/bin/'
  universal_model = SCM({
    "W": lambda n_samples: np.random.normal(size=n_samples),
    "X": linear_model(["W"], [1]),
    "Z": linear_model(["X"], [1]),
    "Y": linear_model(["Z", "W"], [0.2, 0.8])
  })
  universal_model.draw_model()
  print(universal_model.get_parents("Y"))
  print(universal_model.get_endogenous())
  print(universal_model.get_leaves())
