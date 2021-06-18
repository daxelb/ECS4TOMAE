import util
from gutil import permutations, only_given_keys
import gutil
from assignment_models import AssignmentModel, discrete_model, random_model
from environment import Environment
import copy

class Knowledge():
  def __init__(self, environment):
    self.model = environment.cgm
    self.domains = environment.domains
    self.act_vars = environment.act_vars
    self.samples = []
    # self.obs = list()
    # self.exp = list()

  def add_sample(self, sample):
    self.samples.append(sample)

  # def add_sample(self, dataset, sample):
  #   if len(self.model.observed_variables) != len(sample):
  #     print("Error adding sample to agent's dataset.")
  #   # if sample is a list, format correctly as dict
  #   if type(sample) is list:
  #     sample = dict(zip(self.model.observed_variables, sample))
  #   dataset.append(sample)
  
  # def add_obs(self, sample):
  #   self.add_sample(self.obs, sample)
  
  # def add_exp(self, sample):
  #   self.add_sample(self.exp, sample)
  
  # def get_useful_data(self):
  #   """
  #   Returns any data in the model that satisfied do(act_vars)
  #   """
  #   return self.obs + self.exp if self.obs_is_useful() else self.exp

  # def obs_is_useful(self):
  #   for var in self.act_vars:
  #     if self.model.has_latent_parents(var):
  #       return False
  #   return True
  
  def get_useful_data(self):
    return self.samples

  def kl_divergence_of_query(self, query, other_data):
    """
    Returns the KL Divergence of a query between this (self) dataset
    and another dataset (presumably, another agent's useful_data)
    """
    return util.kl_divergence(self.domains, self.get_useful_data(), other_data, query[0], query[1])

  def kl_divergence_of_node(self, node, other_data):
    """
    Returns KL Divergence of the conditional probability of a given node in the model
    between this useful data and another dataset.
    """
    assert node not in self.act_vars
    return self.kl_divergence_of_query(self.model.get_node_dist(node), other_data)

        # shared_items = {k: query2[k] for k in query1 if k in query2 and query1[k] == query2[k]}
        # print(len(shared_items))

if __name__ == "__main__":
  k = Knowledge(Environment({
    "W": random_model((0.5, 0.5)),
    "X": AssignmentModel(("W"), None, (0, 1)),
    "Z": discrete_model(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": discrete_model(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  }))
  print(k.query_as_cpts(({"Y": 1}, {"X": 0})))