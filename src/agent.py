from gutil import permutations, only_dicts_with_givens, Counter, max_key
from query import Product
from util import hash_from_dict, dict_from_hash
import numpy as np
from copy import copy
class Agent:
  def __init__(self, rng, name, environment, databank, div_node_conf=None, asr="EG", epsilon=0, rand_trials=0, cooling_rate=0):
    self.rng = rng
    self.name = name
    self.environment = environment
    self.databank = databank
    self.div_node_conf = div_node_conf
    self.asr = asr
    self.epsilon = epsilon
    self.rand_trials = rand_trials
    self.cooling_rate = cooling_rate
    self.action_var = environment.get_act_var()
    self.action_domain = environment.get_act_dom()
    self.reward_var = environment.get_rew_var()
    self.reward_domain = environment.get_rew_dom()
    self.databank.add_agent(self)
      
  def get_recent(self):
    return self.databank[self][-1]
  
  def get_ind_var_value(self, ind_var):
    if ind_var == "div_node_conf":
      return self.div_node_conf
    elif ind_var == "policy":
      return self.get_policy()
    elif ind_var == "asr":
      return self.asr
    elif ind_var == "epsilon":
      return self.epsilon
    elif ind_var == "rand_trials":
      return self.rand_trials
    elif ind_var == "cooling_rate":
      return self.cooling_rate
    else:
      raise ValueError("Input independent variable is not a property of", str(self))
    
      
  def act(self):
    givens = self.environment.pre.sample(self.rng)
    choice = self.choose(givens)
    givens |= choice
    observation = self.environment.post.sample(self.rng, givens)
    self.databank[self].append(observation)
      
  def choose(self, givens):
    if self.asr == "G":
      return self.choose_optimal(givens)
    elif self.asr == "EG":
      if self.rng.random() < self.epsilon:
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == "EF":
      if self.rand_trials > 0:
        self.rand_trials -= 1
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == "ED":
      if self.rng.random() < self.epsilon:
        self.epsilon *= (1 - self.cooling_rate)
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == "TS":
      return self.thompson_sample(givens)
  
  def choose_optimal(self, givens):
    pass
  
  def choose_random(self):
    return self.rng.choice(permutations(self.action_domain))
  
  def thompson_sample(self, givens):
    pass
  
  def ts_from_dataset(self, dataset, givens):
    choice = None
    max_sample = float('-inf')
    data = dataset.query(givens)
    reward_permutations = permutations(self.reward_domain)
    for action in permutations(self.action_domain):
      alpha = len(data.query({**action, **reward_permutations[1]}))
      beta = len(data.query({**action, **reward_permutations[0]}))
      sample = self.rng.beta(alpha + 1, beta + 1)
      if sample > max_sample:
        choice = action
        max_sample = sample
    return choice
  
  def __copy__(self):
    return self.__class__(self.rng, self.name, self.environment, copy(self.databank), self.div_node_conf, self.asr, self.epsilon, self.rand_trials, self.cooling_rate)
    
  def __hash__(self):
    return hash(self.name)
    
  def __reduce__(self):
    return (self.__class__, (self.rng, self.name, self.environment, self.databank, self.div_node_conf, self.asr, self.epsilon, self.rand_trials, self.cooling_rate))
  
  def __str__(self):
    return self.__class__.__name__ + self.name
  
  def __repr__(self):
    return "<" + self.__class__.__name__ + self.name + ": " + self.asr + ">"
  
  def __eq__(self, other):
    return isinstance(other, self.__class__) \
      and self.name == other.name \
      and self.environment == other.environment \
      and self.asr == other.asr
class SoloAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def get_policy(self):
    return "Solo"
    
  def choose_optimal(self, givens):
    optimal = self.databank[self].optimal_choice(self.action_domain, self.reward_var, givens)
    return optimal if optimal else self.choose_random()
  
  def thompson_sample(self, givens):
    return self.ts_from_dataset(self.databank[self], givens)

class NaiveAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def get_policy(self):
    return "Naive"
    
  def choose_optimal(self, givens):
    optimal = self.databank.all_data().optimal_choice(self.action_domain, self.reward_var, givens)
    return optimal if optimal else self.choose_random()
  
  def thompson_sample(self, givens):
    return self.ts_from_dataset(self.databank.all_data(), givens)

class SensitiveAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def get_policy(self):
    return "Sensitive"
    
  def choose_optimal(self, givens):
    optimal = self.databank.sensitive_data(self).optimal_choice(self.action_domain, self.reward_var, givens)
    return optimal if optimal else self.choose_random()
  
  def thompson_sample(self, givens):
    return self.ts_from_dataset(self.databank.sensitive_data(self), givens)
    
    
class AdjustAgent(SensitiveAgent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.action_var = self.environment.get_act_var()
    
  def get_policy(self):
    return "Adjust"
    
  def div_nodes(self, other):
    return self.databank.div_nodes(self, other)

  def get_num_datapoints(self, tf, other):
    return len(only_dicts_with_givens(self.databank[self], tf[0].get_assignments(tf[0].e())))\
      + len(only_dicts_with_givens(self.databank[other], tf[1].get_assignments(tf[1].e())))
  
  def transport_formula(self, other, givens):
    div_nodes = self.div_nodes(other)
    model = self.environment.cgm
    unformatted_tf = model.from_cpts(
        model.selection_diagram(
          div_nodes
        ).get_transport_formula(
          self.action_var, self.reward_var, set(givens)
        )
      )
    my_queries = Product()
    other_queries = Product()
    for q in unformatted_tf:
      query_var = q.query_var()
      if query_var in givens or query_var == self.action_var:
        continue
      if query_var in div_nodes:
        my_queries.append(q)
      else:
        other_queries.append(q)
    domains = self.environment.domains
    return Product([my_queries, other_queries]).assign(domains).assign(givens)
    
  def solve_transport_formula(self, formula, other):
    formula = formula.deepcopy()
    summation = 0
    for summator in permutations(formula.get_unassigned()):
      formula.assign(summator)
      sol1 = formula[0].solve(self.databank[self])
      sol2 = formula[1].solve(self.databank[other])
      if sol1 is None or sol2 is None:
        return None
      summation += sol1 * sol2
    return summation
    
  def choose_optimal(self, givens):    
    actions = permutations(self.action_domain)
    rewards = permutations(self.reward_domain)
    
    action_rewards = {}
    for action in actions:
      act_hash = hash_from_dict(action)
      action_rewards[act_hash] = {}
      for rew in rewards:
        action_rewards[act_hash][hash_from_dict(rew)] = [[],[]]
    
    for agent in self.databank:
      transport_formula = self.transport_formula(agent, givens)
      for act in actions:
        transport_formula.assign(act)
        act_hash = hash_from_dict(act)
        num_datapoints = self.get_num_datapoints(transport_formula, agent)
        if not num_datapoints:
          continue
        for rew in rewards:
          transport_formula.assign(rew)
          rew_hash = hash_from_dict(rew)
          tf_sol = self.solve_transport_formula(transport_formula, agent)
          if tf_sol is None:
            continue
          action_rewards[act_hash][rew_hash][0].append(num_datapoints)
          action_rewards[act_hash][rew_hash][1].append(tf_sol)
    
    weighted_act_rew = Counter()
    for act in action_rewards:
      for rew in action_rewards[act]:
        reward_prob = 0
        weight_total = sum(action_rewards[act][rew][0])
        if not weight_total: continue
        for i in range(len(action_rewards[act][rew][0])):
          reward_prob += action_rewards[act][rew][1][i] * (action_rewards[act][rew][0][i] / weight_total)
        weighted_act_rew[act] += reward_prob * float(rew.split("=",1)[1])
    optimal = dict_from_hash(max_key(weighted_act_rew))
    return optimal if optimal else self.choose_random()

  
  def thompson_sample(self, givens):
    rewards = permutations(self.reward_domain)
    choice = None
    max_sample = 0
    for action in permutations(self.action_domain):
      alpha_total, beta_total = 0, 0
      for agent in self.databank:
        transport_formula = self.transport_formula(agent, givens)
        transport_formula.assign(action)
        num_datapoints = self.get_num_datapoints(transport_formula, agent)
        if not num_datapoints:
          continue
        transport_formula.assign(rewards[1])
        alpha = self.solve_transport_formula(transport_formula, agent)
        alpha = 0 if alpha is None else alpha * num_datapoints
        beta = num_datapoints - alpha
        alpha_total += alpha
        beta_total += 
        
      sample = self.rng.beta(alpha_total + 1, beta_total + 1)
      if sample > max_sample:
        choice = action
        max_sample = sample
    return choice if choice else self.choose_random()
