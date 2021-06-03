# Besides some added methods, most of this class was written by Iain Barr (ijmbarr on GitHub)
# from his public repository, causalgraphicalmodels
# The code has been imported and modified into this project for ease/consistency

import numpy as np
import inspect
import pandas as pd
import networkx as nx
import os
import util

from scm import StructuralCausalModel
from cgm import CausalGraph
from cam import CausalAssignmentModel, discrete_model


class Environment:
    def __init__(self, assignment):
        """
        Creates StructuralCausalModel from assignment of the form
        { variable: Function(parents) }
        """
        self.assignment = assignment.copy()
        nodes = list(assignment.keys())
        self.action_nodes = []  # set_nodes
        edges = []

        for node, model in assignment.items():
            if model is None:
                # XXX could use a better error type here
                raise ValueError(
                    "Model must be assigned to a CausalAssignmentModel object")

            elif isinstance(model, CausalAssignmentModel):
                # print(model.model, model.parents)
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
                self.assignment[node] = CausalAssignmentModel(parents, model)
                edges.extend([(p, node) for p in parents])

            else:
                raise ValueError("Model must be either callable or None. "
                                 "Instead got {} for node {}."
                                 .format(model, node))

        self.cgm = CausalGraph(
            nodes=nodes, edges=edges, set_nodes=self.action_nodes)

        pre_nodes = []
        [pre_nodes.extend(self.cgm.get_ancestors(v)) for v in self.action_nodes]
        self.pre = StructuralCausalModel(util.only_specified_keys(self.assignment, pre_nodes))
        post_ass = self.assignment
        [post_ass.update({n: CausalAssignmentModel(self.cgm.get_parents(n), None)}) for n in pre_nodes]
        self.post = StructuralCausalModel(post_ass)

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.cgm.dag.nodes())))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))

    def sample(self, set_values=dict()):
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

        # if set_values is None:
        #     set_values = dict()

        for node in nx.topological_sort(self.cgm.dag):
            c_model = self.assignment[node]

            if c_model.model is None:
                samples[node] = set_values[node]
            else:
                parent_samples = {
                    parent: samples[parent]
                    for parent in c_model.parents
                }
                samples[node] = c_model(**parent_samples)

        return samples

    def do(self, node):
        """
        Returns a StructualCausalModel after an intervention on node
        """
        new_assignment = self.assignment.copy()
        new_assignment[node] = None
        return CausalGraph(new_assignment)


if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + \
        'C:/Program Files/Graphviz/bin/'
    universal_model = Environment({
        "W": lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
        "X": CausalAssignmentModel(["W"], None),
        "Z": discrete_model(["X"], {(0,): [0.9, 0.1], (1,): [0.1, 0.9]}),
        "Y": discrete_model(["W", "Z"], {(0, 0): [1, 0], (0, 1): [1, 0], (1, 0): [1, 0], (1, 1): [0, 1]})
    })
#   print(universal_model.sample({"X": 1}))
#   print(universal_model.cgm.get_ancestors("Y"))
    print(universal_model.pre.sample())
    print(universal_model.post.sample(set_values={"W": 1, "X": 1}))
