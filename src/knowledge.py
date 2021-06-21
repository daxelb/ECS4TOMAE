from queries import Quotient, Summation, Product
from query import Query
import util
import gutil
from assignment_models import AssignmentModel, discrete_model, random_model
from environment import Environment

class Knowledge():
  def __init__(self, environment):
    self.model = environment.cgm
    self.domains = environment.domains
    self.act_vars = environment.act_vars
    self.samples = []

  def get_useful_data(self):
    return self.samples
  
  def get_observable(self):
    return sorted(list(self.domains.keys()))
  
  def get_missing(self, dict):
    return [v for v in self.get_observable() if v not in dict.keys()]
  
  def domains_of_missing(self, query):
    return gutil.only_given_keys(self.domains, self.get_missing(query))

  def add_sample(self, sample):
    self.samples.append(sample)

  def get_dist(self, node=None):
    return self.model.get_dist(node).assign(self.domains)
  
  def query_from_cpts(self, query):
    """
    Returns the input query in terms
    of the model's CPTs.
    """
    numerator = Summation(Product([
          q for q in 
          self.get_dist().assign(self.domains).assign(query.get_assignments()) 
          if gutil.first_key(q.Q) in self.model.an(query.Q_and_e().keys())
        ]).over(self.domains_of_missing(query.Q_and_e())))
    denominator = Summation(Product([
          q for q in 
          self.get_dist().assign(self.domains).assign(query.get_assignments()) 
          if gutil.first_key(q.Q) in self.model.an(query.e.keys())
        ]).over(self.domains_of_missing(query.e)))
    return Quotient(numerator, denominator)

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

if __name__ == "__main__":
  k = Knowledge(Environment({
    "W": random_model((0.5, 0.5)),
    "X": AssignmentModel(("W"), None, (0, 1)),
    "Z": discrete_model(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": discrete_model(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  }))
  # print(k.get_dist())
  # print(k.get_distribution("Y"))
  print(k.query_from_cpts(Query({"Y": 0}, ["X"])))