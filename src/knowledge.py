import util

class Knowledge():
  def __init__(self, model, domains, action_vars):
    self.model = model
    self.domains = domains
    self.action_vars = action_vars
    self.obs = list()
    self.exp = list()

  def add_sample(self, dataset, sample):
    if len(self.model.observed_variables) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(self.model.observed_variables, sample))
    dataset.append(sample)
  
  def add_obs(self, sample):
    self.add_sample(self.obs, sample)
  
  def add_exp(self, sample):
    self.add_sample(self.exp, sample)

  def get_useful_data(self):
    """
    Returns any data in the model that satisfied do(action_vars)
    """
    return self.obs + self.exp if self.obs_is_useful() else self.exp

  def obs_is_useful(self):
    for var in self.action_vars:
      if self.model.has_latent_parents(var):
        return False
    return True

  def get_model_dist(self):
    """
    Returns a parsed version of the model's joint prob. dist.
    """
    return util.parse_query(self.model.get_distribution())

  def get_node_dist(self, node):
    """
    Get's the conditional probability distribution
    for a node in the model.
    Ex: A -> B <- C
    => P(B|A,C) (but in the standard format of
    the program)
    """
    for dist in self.get_model_dist():
      if node in dist[0].keys():
        return dist

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
    if node in self.action_vars:
      print("S-nodes cannot be inputs into action nodes.")
    return self.kl_divergence_of_query(self.get_node_dist(node), other_data)
