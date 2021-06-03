# Besides some added methods, most of this class was written by Iain Barr (ijmbarr on GitHub)
# from his public repository, causalgraphicalmodels
# The code has been imported and modified into this project for ease/consistency

import numpy as np
import inspect
import networkx as nx
import os
import util

from scm import StructuralCausalModel
from cgm import CausalGraph
from cam import CausalAssignmentModel, discrete_model


class Environment:
    def __init__(self, domains, assignment):
        """
        Creates StructuralCausalModel from assignment of the form
        { variable: Function(parents) }
        """
        self.domains = domains
        self._assignment = assignment.copy()
        nodes = list(assignment.keys())
        self.action_nodes = []
        edges = []

        for node, model in assignment.items():
            if model is None:
                # XXX could use a better error type here
                raise ValueError(
                    "Model must be assigned to a CausalAssignmentModel object")

            elif isinstance(model, CausalAssignmentModel):
                if model.model == None:
                    self.action_nodes.append(node)
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
                self._assignment[node] = CausalAssignmentModel(parents, model)
                edges.extend([(p, node) for p in parents])

            else:
                raise ValueError("Model must be either callable or None. "
                                 "Instead got {} for node {}."
                                 .format(model, node))

        self.cgm = CausalGraph(edges)

        pre_nodes = []
        [pre_nodes.extend(self.cgm.get_ancestors(v)) for v in self.action_nodes]
        self.pre = StructuralCausalModel(util.only_specified_keys(self._assignment, pre_nodes))
        post_ass = self._assignment.copy()
        [post_ass.update({n: CausalAssignmentModel(self.cgm.get_parents(n), None)}) for n in pre_nodes]
        self.post = StructuralCausalModel(post_ass)

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.cgm.dag.nodes())))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))

if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + \
        'C:/Program Files/Graphviz/bin/'
    domains = {"W": (0,1), "X": (0,1), "Z": (0,1), "Y": (0,1)}
    universal_model = Environment(domains, {
        "W": lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
        "X": CausalAssignmentModel(["W"], None),
        "Z": discrete_model(["X"], {(0,): [0.9, 0.1], (1,): [0.1, 0.9]}),
        "Y": discrete_model(["W", "Z"], {(0, 0): [1, 0], (0, 1): [1, 0], (1, 0): [1, 0], (1, 1): [0, 1]})
    })
#   print(universal_model.sample({"X": 1}))
#   print(universal_model.cgm.get_ancestors("Y"))
    print(universal_model.pre.sample())
    print(universal_model.post.sample(set_values={"W": 1, "X": 1}))
    print(universal_model._assignment["W"].model)
