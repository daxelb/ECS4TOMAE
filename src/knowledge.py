from query import Quotient, Summation, Product, Queries, Query
import util
import gutil
import random

DIV_NODE_CONF = 0.06
SAMPS_NEEDED = 25
# DIV_EPS_DEC_SLOWNESS = 2

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
  
  def domains_of_missing(self, dict):
    return gutil.only_given_keys(self.domains, self.get_missing(dict))

  def add_sample(self, agent, sample):
    if agent not in self.samples:
      self.samples[agent] = []
    self.samples[agent].append(sample)

  def get_dist(self, node=None):
    return self.model.get_dist(node).assign(self.domains)
  
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
    
  def add_sample(self, agent, sample):
    super().add_sample(agent, sample)
    self.update_divergence(agent)

  def div_nodes(self, agent):
    return [node for node, divergence in self.divergence[agent].items() if divergence is None or abs(divergence) > DIV_NODE_CONF]
    
  def sensitive_data(self):
    data = []
    for agent, agent_data in self.samples.items():
      if not self.div_nodes(agent):
        data.extend(agent_data)
    return data
    
  def kl_divergence_of_query(self, query, other_data):
    """
    Returns the KL Divergence of a query between this (self) dataset
    and another dataset (presumably, another agent's useful_data)
    """
    
    return util.kl_divergence(self.domains, self.my_data(), other_data, query.Q, query.e)

  def kl_divergence_of_node(self, node, other_data):
    """
    Returns KL Divergence of the conditional probability of a given node in the model
    between this useful data and another dataset.
    """
    assert node not in self.act_vars
    return self.kl_divergence_of_query(self.model.get_node_dist(node), other_data)

  def get_non_action_nodes(self):
    return [node for node in self.model.get_observable() if node not in self.act_vars]

  def update_divergence(self, agent):
    if agent not in self.divergence:
      self.add_agent_divergence(agent)
    if len(self.samples[agent]) < SAMPS_NEEDED or agent == self.agent:
      return
    for node in self.get_non_action_nodes():
      self.divergence[agent][node] = self.kl_divergence_of_node(node, self.samples[agent])

  def add_agent_divergence(self, agent):
    self.divergence[agent] = {}
    if agent == self.agent:
      self.assign_divergence(agent, 0)
    else:
      self.assign_divergence(agent, 1)
      
  def assign_divergence(self, agent, assignment):
    for node in self.get_non_action_nodes():
      self.divergence[agent][node] = assignment
      
  def is_divergent_dict(self, agent):
    divergent = {}
    for node in self.divergence[agent]:
      node_divergence = self.divergence[agent][node]
      divergent[node] = node_divergence is None or abs(node_divergence) > DIV_NODE_CONF
    return divergent
      
  def optimal_choice(self, givens={}):
    expected_values = util.expected_vals(self.sensitive_data(), self.act_vars, self.rew_var, givens)
    return util.dict_from_hash(gutil.max_key(expected_values)) if expected_values else None

class KnowledgeAdjust(KnowledgeSensitive):
  def __init__(self, *args):
    super().__init__(*args) 
  
  def transport_formula(self, agent, givens):
    unformatted_tf = self.model.from_cpts(
        self.model.selection_diagram(
          self.div_nodes(agent)
        ).get_transport_formula(
          self.act_vars[0], self.rew_var, set(givens)
        )
      )
    my_queries = Product()
    other_queries = Product()
    div_nodes = self.div_nodes(agent)
    for q in unformatted_tf:
      query_var = q.query_var()
      if query_var in givens or query_var in self.act_vars:
        continue
      if query_var in div_nodes:
        my_queries.append(q)
      else:
        other_queries.append(q)
    return Product([my_queries, other_queries]).assign(self.domains).assign(givens)
  
  def from_cpts(self, query, givens):
    """
    Returns the input query in terms
    of the model's CPTs.
    """
    return Product([
        q for q in
        self.model.get_dist().assign(self.domains).assign(givens)
        if q.query_var() not in givens and
        q.contains(query.get_vars())
      ])
    
  def formatted_transport_formula(self, agent, givens, action):
    div_nodes = self.div_nodes(agent)
    my_tf = Product()
    other_tf = Product()
    for q in self.from_cpts(self.transport_formula(agent, givens), {**action, **givens}):
      if q.query_var() in div_nodes:
        my_tf.append(q)
        continue
      other_tf.append(q)
    return Product([my_tf, other_tf])
    
  def solve_transport_formula(self, f, my_data, other_data):
    f = f.deepcopy()
    summation = 0
    for summator in gutil.permutations(f.get_unassigned()):
      f.assign(summator)
      sol1 = f[0].solve(my_data)
      sol2 = f[1].solve(other_data)
      if sol1 is None or sol2 is None:
        return None
      summation += sol1 * sol2
    return summation
  
  def get_uncomputed_prob_of_tf(self, tf, my_data, other_data):
    f = tf.deepcopy()
    summation = [0,0]
    for summator in gutil.permutations(f.get_unassigned()):
      f.assign(summator)
      my_prob = f[0].uncomputed_prob(my_data)
      other_prob = f[1].uncomputed_prob(other_data)
      if my_prob is None or other_prob is None:
        return None
      summation[0] += my_prob[0] * other_prob[0]
      summation[1] += my_prob[1] * other_prob[1]
    return summation

  def get_num_datapoints(self, tf, other):
    return len(gutil.only_dicts_with_givens(self.my_data(), tf[0].get_assignments(tf[0].e())))\
      + len(gutil.only_dicts_with_givens(self.samples[other], tf[1].get_assignments(tf[1].e())))
    
    
  def optimal_choice(self, givens={}):
    if len(self.my_data()) < SAMPS_NEEDED:
      return Knowledge.optimal_choice(self, givens)
    
    actions = gutil.permutations(self.act_doms)
    rewards = gutil.permutations(self.rew_dom)
    my_data = self.my_data()
    
    action_rewards = {}
    for action in actions:
      act_hash = util.hash_from_dict(action)
      action_rewards[act_hash] = {}
      for rew in rewards:
        action_rewards[act_hash][util.hash_from_dict(rew)] = [[],[]]
        # action_rewards[act_hash][util.hash_from_dict(rew)] = [0,0]
    
    for other, other_data in self.samples.items():
      transport_formula = self.transport_formula(other, givens)
      for action in actions:
        transport_formula.assign(action)
        act_hash = util.hash_from_dict(action)
        
        # this is only true for the model we are using with vars W,X,Y,Z and does not always hold.
        # the formula should be: number of datapoints where conditionals are satisfied for each
        # part of the transport formula.
        # Ex: P*(A|B=1) * P(C|A) would be: num*(data B=1) + num(all data)
        # num_datapoints = len(gutil.only_dicts_with_givens(data, givens)) + len(gutil.only_dicts_with_givens(my_data, action))
        
        num_datapoints = self.get_num_datapoints(transport_formula, other)
        if not num_datapoints:
          continue
        # calculating the transport formula with every action and every agent is slow
        # transport_formula = self.formatted_transport_formula(agent, givens, action)
        for reward in rewards:
          rew_hash = util.hash_from_dict(reward)
          transport_formula.assign(reward)
          tf_sol = self.solve_transport_formula(transport_formula, my_data, other_data)
          if tf_sol is None:
            continue
          action_rewards[act_hash][rew_hash][0].append(num_datapoints)
          action_rewards[act_hash][rew_hash][1].append(tf_sol)
          # action_rewards[act_hash][0].append(gutil.first_value(reward) * tf_sol[0])
          # action_rewards[act_hash][1] += tf_sol[1]
    
    # for act_key in action_rewards:
    #   denom = action_rewards[act_key][1]
    #   action_rewards[act_key] = sum(action_rewards[act_key][0]) / denom if denom else None
    
    weighted_act_rew = gutil.Counter()
    for act in action_rewards:
      for rew in action_rewards[act]:
        reward_prob = 0
        weight_total = sum(action_rewards[act][rew][0])
        if not weight_total: continue
        for i in range(len(action_rewards[act][rew][0])):
          reward_prob += action_rewards[act][rew][1][i] * (action_rewards[act][rew][0][i] / weight_total)
        weighted_act_rew[act] += reward_prob * float(rew.split("=",1)[1])
    return util.dict_from_hash(gutil.max_key(weighted_act_rew))