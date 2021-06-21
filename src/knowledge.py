from queries import Quotient, Summation, Product
import util
import gutil
import random

DIV_NODE_CONF = 0.075
SAMPS_NEEDED = 20
DIV_EPS_DEC_SLOWNESS = 1.75

class Knowledge():
  def __init__(self, environment, agent):
    self.model = environment.cgm
    self.domains = environment.domains
    self.act_vars = environment.act_vars
    self.rew_var = environment.rew_var
    self.agent = agent
    self.samples = {agent: []}
    self.act_doms = gutil.only_given_keys(self.domains, self.act_vars)
    self.rew_dom = gutil.only_given_keys(self.domains, [self.rew_var])

  def my_data(self):
    return self.samples[self.agent]
  
  def get_observable(self):
    return sorted(list(self.domains.keys()))
  
  def get_missing(self, dict):
    return [v for v in self.get_observable() if v not in dict.keys()]
  
  def domains_of_missing(self, query):
    return gutil.only_given_keys(self.domains, self.get_missing(query))

  def add_sample(self, agent, sample):
    if agent not in self.samples:
      self.samples[agent] = []
    self.samples[agent].append(sample)

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
  
  def optimal_choice(self, givens={}):
    expected_values = util.expected_vals(self.my_data(), self.act_vars, self.rew_var, givens)
    return util.dict_from_hash(gutil.max_key(expected_values)) if expected_values else None
class KnowledgeNaive(Knowledge):
  def __init__(self, *args):
    super().__init__(*args)
    
  def all_data(self):
    data = []
    for a in self.samples:
      data.extend(self.samples[a])
    return data
  
  def optimal_choice(self, givens={}):
    expected_values = util.expected_vals(self.all_data(), self.act_vars, self.rew_var, givens)
    return util.dict_from_hash(gutil.max_key(expected_values)) if expected_values else None
  
class KnowledgeSensitive(Knowledge):
  def __init__(self, *args):
    super().__init__(*args)
    self.divergence = {self.agent: gutil.Counter()}
    
  def sensitive_data(self):
    data = []
    for a in self.divergence:
      if not self.div_nodes(a):
        data.extend(self.samples[a])
    return data

  def add_sample(self, agent, sample):
    super().add_sample(agent, sample)
    if agent not in self.divergence:
      self.divergence[agent] = gutil.Counter()
    self.update_divergence(agent)
    
  def kl_divergence_of_query(self, query, other_data):
    """
    Returns the KL Divergence of a query between this (self) dataset
    and another dataset (presumably, another agent's useful_data)
    """
    return util.kl_divergence(self.domains, self.my_data(), other_data, query[0], query[1])

  def kl_divergence_of_node(self, node, other_data):
    """
    Returns KL Divergence of the conditional probability of a given node in the model
    between this useful data and another dataset.
    """
    assert node not in self.act_vars
    return self.kl_divergence_of_query(self.model.get_node_dist(node), other_data)

  def update_divergence(self, agent):
    div_epsilon = (SAMPS_NEEDED * DIV_EPS_DEC_SLOWNESS)/(len(self.samples[agent]) - SAMPS_NEEDED + SAMPS_NEEDED * DIV_EPS_DEC_SLOWNESS)
    for node in self.divergence[agent]:
      if random.random() >= div_epsilon and self.is_divergent_dict(agent)[node] == False:
        continue
      self.divergence[agent][node] = self.knowledge.kl_divergence_of_node(node, self.samples[agent])
      
  def is_divergent_dict(self, agent):
    divergent = {}
    for node in self.divergence[agent]:
      divergent[node] = self.divergence[agent][node] < DIV_NODE_CONF
    return divergent
    
  def div_nodes(self, agent):
    return [node for node, divergence in self.divergence[agent].items() if divergence > DIV_NODE_CONF]
      
  def optimal_choice(self, givens={}):
    expected_values = util.expected_vals(self.sensitive_data(), self.act_vars, self.rew_var, givens)
    return util.dict_from_hash(gutil.max_key(expected_values)) if expected_values else None

class KnowledgeAdjust(KnowledgeSensitive):
  def __init__(self, *args):
    super().__init__(*args) 

  def selection_diagram(self, agent):
    return self.model.selection_diagram(self.div_nodes(agent))
  
  def transport_formula(self, agent, givens):
    return self.selection_diagram(agent).get_transport_formula(self.act_vars[0], self.rew_var, list(givens)).assign(self.domains).assign(givens)
    
  def optimal_choice(self, givens={}):
    if len(self.my_data()) < SAMPS_NEEDED:
      return Knowledge.optimal_choice(self, givens)
    
    action_rewards = {}
    for action in gutil.permutations(self.act_doms):
      act_hash = util.hash_from_dict(action)
      action_rewards[act_hash] = {}
      for rew in gutil.permutations(self.rew_dom):
        action_rewards[act_hash][util.hash_from_dict(rew)] = [[],[]]
    
    for agent, data in self.samples.items():
      if len(data) < SAMPS_NEEDED:
        continue
      transport_formula = self.transport_formula(agent, givens)
      if not transport_formula:
        continue
      for action in gutil.permutations(self.act_doms):
        act_hash = util.hash_from_dict(action)
        num_datapoints = len(gutil.only_dicts_with_givens(data, {**action, **givens}))
        if not num_datapoints:
          continue
        for rew in gutil.permutations(self.rew_dom):
          rew_hash = util.hash_from_dict(rew)
          transport_formula.assign({**rew, **action})
          action_rewards[act_hash][rew_hash][0].append(num_datapoints)
          action_rewards[act_hash][rew_hash][1].append(transport_formula.solve(data))
    
    weighted_act_rew = gutil.Counter()
    for act in action_rewards:
      for rew in action_rewards[act]:
        reward_prob = 0
        weight_total = sum(action_rewards[act][rew][0])
        if not weight_total: continue
        for i in range(len(action_rewards[act][rew][0])):
          reward_prob += action_rewards[act][rew][1][i] * (action_rewards[act][rew][0][i] / weight_total)
        weighted_act_rew[act] += reward_prob * float(rew.split("=",1)[1])
    return util.dict_from_hash(gutil.max_key(weighted_act_rew)) if weighted_act_rew else None