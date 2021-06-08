# Besides some added methods, most of this class was written by Iain Barr (ijmbarr on GitHub)
# from his public repository, causalgraphicalmodels
# The code has been imported and modified into this project for ease/consistency

import numpy as np
import inspect
import networkx as nx
import os
import util
import math

from scm import StructuralCausalModel
from cgm import CausalGraph
from assignment_models import AssignmentModel, discrete_model, random_model


class Environment:
    def __init__(self, assignment, reward_node="Y"):
        """
        Creates StructuralCausalModel from assignment of the form
        { variable: Function(parents) }
        """
        self.domains = {}
        self._assignment = assignment.copy()
        nodes = list(assignment.keys())
        self.action_nodes = []
        self.reward_node = reward_node
        set_nodes = []
        edges = []

        for node, model in assignment.items():
            if model is None:
                set_nodes.append(node)

            elif isinstance(model, AssignmentModel):
                self.domains[node] = model.domain
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
                self._assignment[node] = AssignmentModel(parents, model)
                edges.extend([(p, node) for p in parents])

            else:
                raise ValueError("Model must be either callable or None. "
                                 "Instead got {} for node {}."
                                 .format(model, node))

        self.cgm = CausalGraph(nodes=nodes, edges=edges, set_nodes=set_nodes)

        pre_nodes = []
        [pre_nodes.extend(self.cgm.get_ancestors(v)) for v in self.action_nodes]
        self.pre = StructuralCausalModel(util.only_specified_keys(self._assignment, pre_nodes))
        post_ass = self._assignment.copy()
        [post_ass.update({n: AssignmentModel(self.cgm.get_parents(n), None, self.domains[n])}) for n in pre_nodes]
        self.post = StructuralCausalModel(post_ass)
        self.action_rewards = self.get_action_rewards()
        
    def get_action_rewards(self, iterations=200):
      act_feat_nodes = [n for n in self.action_nodes]
      [act_feat_nodes.extend(self.cgm.get_parents(a)) for a in self.action_nodes]
      act_feat_nodes = util.remove_dupes(act_feat_nodes)
      perms = util.permutations(util.only_specified_keys(self.domains, act_feat_nodes))
      action_rewards = []
      for p in perms:
        action_reward = [p,0]
        for _ in range(iterations):
          action_reward[1] += self.post.sample(p)[self.reward_node]
        action_reward[1] /= iterations
        action_rewards.append(tuple(action_reward))
      return action_rewards
    
    def optimal_action_rewards(self, givens={}):
      action_rewards = []
      for tup in self.action_rewards:
        action_rewards.append((util.only_specified_keys(tup[0], self.action_nodes), tup[1]))
        for key in givens:
          if tup[0][key] != givens[key]:
            action_rewards = action_rewards[:-1]
            break
      best = []
      best_rew = -math.inf
      for tup in action_rewards:
        if tup[1] > best_rew:
          best_rew = tup[1]
          best = [tup]
        elif tup[1] == best_rew:
          best.append(tup)
      return best
    
    def optimal_actions(self, givens={}):
      return [tup[0] for tup in self.optimal_action_rewards(givens)]
    
    def optimal_rewards(self, givens={}):
      return self.optimal_action_rewards(givens)[0][1]

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.cgm.dag.nodes())))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))

    def __hash__(self):
        return hash(util.dict_to_list_of_tuples(self._assignment))

    def __eq__(self, other):
        for key in self._assignment:
            if self._assignment[key] != other._assignment[key]:
                return False
        return True

if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + \
        'C:/Program Files/Graphviz/bin/'
    domains = {"W": (0,1), "X": (0,1), "Z": (0,1), "Y": (0,1)}
    universal_model = Environment({
    "W": random_model((0.5, 0.5)),
    "X": AssignmentModel(("W"), None, (0, 1)),
    "Z": discrete_model(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": discrete_model(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  })
    # print(universal_model.sample({"X": 1}))
    # print(universal_model.cgm.get_ancestors("Y"))
    # print(universal_model.pre.sample())
    # print(universal_model.post.sample(set_values={"W": 1, "X": 1}))
    # print(universal_model._assignment["W"].model)
    # print(util.only_dicts_with_specified_entries(universal_model.get_action_rewards2(), {"X": 1}))
    print(universal_model.optimal_rewards({"W": 0}))
    # print(universal_model.optimal_act_rew({"W":1}))
