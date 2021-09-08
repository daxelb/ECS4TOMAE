from query import Product, Query
from util import permutations, only_dicts_with_givens
from data import DataSet
from enums import ASR

class Agent:
  def __init__(self, rng, name, environment, databank, tau=None, asr="EG", epsilon=0, rand_trials=0, cooling_rate=0):
    self.rng = rng
    self.name = name
    self.environment = environment
    self.databank = databank
    self.tau = tau
    self.asr = asr
    self.feat_perms = permutations(environment.get_feat_doms())
    self.epsilon = [1] * len(self.feat_perms) if asr == ASR.ED else epsilon
    self.rand_trials = rand_trials
    self.rand_trials_rem = [rand_trials] * len(self.feat_perms)
    self.cooling_rate = cooling_rate
    self.action_var = environment.get_act_var()
    self.action_domain = environment.get_act_dom()
    self.reward_var = environment.get_rew_var()
    self.reward_domain = environment.get_rew_dom()
    
    self.databank.add_agent(self)
      
  def get_recent(self):
    return self.databank[self][-1]
  
  def get_ind_var_value(self, ind_var):
    if ind_var == "tau":
      return self.tau
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
      return ""
      
  def act(self):
    givens = self.environment.pre.sample(self.rng)
    choice = self.choose(givens)
    givens |= choice
    observation = self.environment.post.sample(self.rng, givens)
    self.databank[self].append(observation)
      
  def choose(self, givens):
    if self.asr == ASR.G:
      return self.choose_optimal(givens)
    elif self.asr == ASR.EG:
      if self.rng.random() < self.epsilon:
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == ASR.EF:
      given_i = self.feat_perms.index(givens)
      if self.rand_trials_rem[given_i] > 0:
        self.rand_trials_rem[given_i] -= 1
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == ASR.ED:
      given_i = self.feat_perms.index(givens)
      if self.rng.random() < self.epsilon[given_i]:
        self.epsilon[given_i] *= self.cooling_rate
        return self.choose_random()
      self.epsilon[given_i] *= self.cooling_rate
      return self.choose_optimal(givens)
    elif self.asr == ASR.TS:
      return self.thompson_sample(givens)
  
  def choose_optimal(self, givens):
    pass
  
  def choose_random(self):
    return self.rng.choice(permutations(self.action_domain))
  
  def thompson_sample(self, givens):
    pass
  
  def ts_from_dataset(self, dataset, givens):
    choice = None
    max_sample = 0 #float('-inf')
    data = dataset.query(givens)
    for action in permutations(self.action_domain):
      alpha = len(data.query({**action, **{self.reward_var: 1}}))
      beta = len(data.query({**action, **{self.reward_var: 0}}))
      sample = self.rng.beta(alpha + 1, beta + 1)
      if sample > max_sample:
        choice = action
        max_sample = sample
    return choice
  
  def get_policy(self):
    return self.__class__.__name__[:-5]

  def __hash__(self):
    return hash(self.name)
    
  def __reduce__(self):
    return (self.__class__, (self.rng, self.name, self.environment, self.databank, self.tau, self.asr, self.epsilon, self.rand_trials, self.cooling_rate))
  
  def __repr__(self):
    return "<" + self.get_policy() + self.name + ": " + self.asr.value + ">"
  
  def __eq__(self, other):
    return isinstance(other, self.__class__) \
      and self.name == other.name

class SoloAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def choose_optimal(self, givens):
    optimal = self.databank[self].optimal_choice(self.rng, self.action_domain, self.reward_var, givens)
    return optimal if optimal else self.choose_random()
  
  def thompson_sample(self, givens):
    return self.ts_from_dataset(self.databank[self], givens)

class NaiveAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def choose_optimal(self, givens):
    optimal = self.databank[self].optimal_choice(self.rng, self.action_domain, self.reward_var, givens)
    return optimal if optimal else self.choose_random()
  
  def thompson_sample(self, givens):
    return self.ts_from_dataset(self.databank.all_data(), givens)

class SensitiveAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def choose_optimal(self, givens):
    optimal = self.databank.sensitive_data(self).optimal_choice(self.rng, self.action_domain, self.reward_var, givens)
    return optimal if optimal else self.choose_random()
  
  def thompson_sample(self, givens):
    return self.ts_from_dataset(self.databank.sensitive_data(self), givens)
    
class AdjustAgent(SensitiveAgent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.action_var = self.environment.get_act_var()
    
  def has_S_node(self, node, other):
    return node in self.div_nodes(other)

  def div_nodes(self, other):
    return self.databank.div_nodes(self, other)

  def get_num_datapoints(self, tf, other):
    return len(only_dicts_with_givens(self.databank[self], tf[0].get_assignments(tf[0].e())))\
      + len(only_dicts_with_givens(self.databank[other], tf[1].get_assignments(tf[1].e())))
  
  def transport_formula(self, div_nodes, givens):
    model = self.environment.cgm
    unformatted_tf = model.from_cpts(
        model.selection_diagram(
          div_nodes
        ).get_transport_formula(
          self.action_var, self.reward_var, set(givens)
        )
      )
    query = Product()
    for q in unformatted_tf:
      query_var = q.query_var()
      if query_var not in givens and query_var != self.action_var:
        query.append(q)
    return query.assign(self.environment.domains).assign(givens)

  def get_CPTs(self):
    div_nodes = {a: self.div_nodes(a) for a in self.databank}
    CPTs = {}
    for node in self.environment.get_non_act_vars():
      CPTs[node] = DataSet()
      for agent, data in self.databank.items():
        if node not in div_nodes[agent]:
          CPTs[node].extend(data)
    return CPTs

  def choose_optimal(self, givens):
    CPTs = self.get_CPTs()
    max_val = 0
    choices = []
    for action in permutations(self.action_domain):
      expected_value = self.get_alpha_beta(CPTs, action, givens)[0]
      if expected_value > max_val:
        max_val = expected_value
        choices = [action]
      elif expected_value == max_val:
        choices.append(action)
    return self.rng.choice(choices)

  def thompson_sample(self, givens):
    CPTs = self.get_CPTs()
    max_sample = 0
    choices = []
    for action in permutations(self.action_domain):
      alpha_beta = self.get_alpha_beta(CPTs, action, givens)
      sample = self.rng.beta(alpha_beta[0] + 1, alpha_beta[1] + 1)
      if sample > max_sample:
        max_sample = sample
        choices = [action]
      if sample == max_sample:
        choices.append(action)
    return self.rng.choice(choices)


  def get_alpha_beta(self, CPTs, action, givens):
    weight = len(CPTs["Y"].query(givens)) + len(CPTs["W"].query({**givens, **action}))
    alpha_prob = self.get_prob_reward(CPTs, action, givens, 1)
    if alpha_prob is None:
      return (0,0)
    return (alpha_prob * weight, (1-alpha_prob) * weight)

  def get_prob_reward(self, CPTs, action, givens, rew_assignment):
    prob = 0
    for w in (0,1):
      y_prob = Query({"Y": rew_assignment}, {**{"W": w}, **givens}).solve(CPTs["Y"])
      w_prob = Query({"W": w}, action).solve(CPTs["W"])
      if y_prob is None or w_prob is None:
        return None
      prob += y_prob * w_prob
    return prob
      

  # def get_alpha_beta(self, CPTs, div_nodes, action, givens):
  #   alpha, beta = 1, 1
  #   for w in (0,1):
  #     weight = len(CPTs["Y"].query({**{"W": w},**givens})) + len(CPTs["W"].query({**{"W":w}, **givens, **action}))
  #     pt1 = Query({"Y": 1}, {**{"W": w}, **givens}).solve(CPTs["Y"])
  #     pt2 = Query({"W": w}, action).solve(CPTs["W"])
  #     if pt1 is None or pt2 is None:
  #       return (1,1)
  #     alpha += pt1 * pt2 * weight
  #     beta += (1-(pt1 * pt2)) * weight
  #   return (alpha, beta)





    #Query({"Y": 1}, givens).solve(CPTs["Y"]) * Query({"W": (0,1)}, action).solve(CPTs["W"])
    # tf = (Query({"Y": 1}, {"W": (0,1), **givens}), Query({"W": (0,1)}, action))
    # weight = CPTs["Y"].num(givens) + CPTs["W"].num({**givens, **action})
    # weight = CPTs["Y"].num(givens) + CPTs["W"].num({**action})
    # tf.assign({"Y": 1})
    
    # sol = 0 if tf[0].solve(CPTs["Y"]) is None or tf[1].solve(CPTs["W"]) is None else tf[0].solve(CPTs["Y"]) * tf[1].solve(CPTs["W"])
    # sol = Product([q.solve(CPTs[q.query_var()])
    #                 for q in tf]).solve()
    # alpha = sol
    # tf = (Query({"Y": 1}, {"W": (0, 1), **givens}),
    #       Query({"W": (0, 1)}, action))
    # sol = tf[0].solve(CPTs["Y"]) * tf[1].solve(CPTs["W"])
    # beta = 0 if sol is None else sol
    # return (alpha, 0)#(1 + (alpha * weight), 1 + (beta * weight))

  # def get_alpha_beta(self, CPTs, div_nodes, action, givens):
  #   alpha, beta = 1,1
  #   for agent in self.databank.keys():
  #     if div_nodes[agent].issubset(self.environment.get_feat_vars()):
  #       tf = self.transport_formula(div_nodes[agent], givens)
  #       tf.assign(action)
  #       weight = self.databank[agent].num({**action, **givens})
  #       tf.assign({"Y": 1})
  #       sol = Product([q.solve(CPTs[q.query_var()])
  #                      for q in tf]).solve()
  #       alpha += 0 if sol is None else sol * weight
  #       tf.assign({"Y": 0})
  #       sol = Product([q.solve(CPTs[q.query_var()])
  #                      for q in tf]).solve()
  #       beta += 0 if sol is None else sol * weight
  #     elif "Y" not in div_nodes[agent]:
  #       tf = Query({"Y":(0,1)},{"W":(0,1), **givens}) #self.transport_formula(div_nodes[agent], givens)
  #       weight = self.databank[agent].num({**givens})
  #       tf.assign({"Y": 1})
  #       sol = tf.solve(CPTs["Y"])
  #       alpha += 0 if sol is None else sol * weight
  #       tf.assign({"Y": 1})
  #       sol = tf.solve(CPTs["Y"])
  #       beta += 0 if sol is None else sol * weight
  #   return (alpha, beta)