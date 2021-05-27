import numpy as np
from causalgraphicalmodels.csm import StructuralCausalModel, linear_model, logistic_model
from util import alphabetize

class SCM(StructuralCausalModel):
  def __init__(self, assignment):
    super(SCM, self).__init__(assignment)

  def get_variables(self):
    return alphabetize(list(self.cgm.dag.nodes))

  def get_parents(self, node):
    return alphabetize(list(self.cgm.dag.predecessors(node)))

  def get_children(self, node):
    return alphabetize(list(self.cgm.dag.successors(node)))

  def get_exogenous(self):
    return alphabetize([n for n in self.cgm.dag.nodes if not self.get_parents(n)])

  def get_endogenous(self):
    return alphabetize([n for n in self.cgm.dag.nodes if self.get_parents(n)])

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

if __name__ == "__main__":
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