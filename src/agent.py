from query import Product, Query
from util import hash_from_dict, dict_from_hash, permutations, only_dicts_with_givens, max_key, Counter
from data import DataSet
class Agent:
  def __init__(self, rng, name, environment, databank, div_node_conf=None, asr="EG", epsilon=0, rand_trials=0, cooling_rate=0):
    self.rng = rng
    self.name = name
    self.environment = environment
    self.databank = databank
    self.div_node_conf = div_node_conf
    self.asr = asr
    self.epsilon = 1 if asr == "ED" else epsilon
    self.rand_trials = rand_trials
    self.rand_trials_rem = rand_trials
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
      return ""
      
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
      if self.rand_trials_rem > 0:
        self.rand_trials_rem -= 1
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == "ED":
      if self.rng.random() < self.epsilon:
        self.epsilon *= self.cooling_rate
        return self.choose_random()
      self.epsilon *= self.cooling_rate
      return self.choose_optimal(givens)
    elif self.asr == "TS":
      return self.thompson_sample(givens)
  
  def choose_optimal(self, givens):
    pass
  
  def choose_random(self):
    # print("@!")
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
    return (self.__class__, (self.rng, self.name, self.environment, self.databank, self.div_node_conf, self.asr, self.epsilon, self.rand_trials, self.cooling_rate))
  
  def __repr__(self):
    return "<" + self.get_policy() + self.name + ": " + self.asr + ">"
  
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


  # def transport_formula(self, other, givens):
  #   div_nodes = self.div_nodes(other)
  #   model = self.environment.cgm
  #   unformatted_tf = model.from_cpts(
  #       model.selection_diagram(
  #         div_nodes
  #       ).get_transport_formula(
  #         self.action_var, self.reward_var, set(givens)
  #       )
  #     )
  #   my_queries = Product()
  #   other_queries = Product()
  #   for q in unformatted_tf:
  #     query_var = q.query_var()
  #     if query_var in givens or query_var == self.action_var:
  #       continue
  #     if query_var in div_nodes:
  #       my_queries.append(q)
  #     else:
  #       other_queries.append(q)
  #   return Product([my_queries, other_queries]).assign(self.environment.domains).assign(givens)
    
  def solve_transport_formula(self, tf, other):
    tf = tf.deepcopy()
    summation = 0
    for summator in permutations(tf.get_unassigned()):
      tf.assign(summator)
      sol1 = tf[0].solve(self.databank[self])
      sol2 = tf[1].solve(self.databank[other])
      if sol1 is None or sol2 is None:
        return None
      summation += sol1 * sol2
    return summation

  def solve_transport_formula_new(self, tf, other):
    num_datapoints = len(only_dicts_with_givens(self.databank[self], tf[0].get_assignments(tf[0].e()))) + \
                     len(only_dicts_with_givens(self.databank[other], tf[1].get_assignments(tf[1].e())))
    tf = tf.deepcopy()
    summation = 0
    # b_summation = 0
    for summator in permutations(tf.get_unassigned()):
      tf.assign(summator)
      tf0 = tf[0].solve(self.databank[self])
      tf1 = tf[1].solve(self.databank[other])
      if tf0 is None or tf1 is None:
        return None
      a_sol1 = tf0 #* num_datapoints[0]
      # b_sol1 = num_datapoints[0] - a_sol1
      a_sol2 = tf1 #* num_datapoints[1]
      # b_sol2 = num_datapoints[1] - a_sol2
      summation += a_sol1 * a_sol2
      # b_summation += b_sol1 * b_sol2
    return (num_datapoints * summation, num_datapoints - (num_datapoints * summation))
    
  def choose_optimal2(self, givens):
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
        if not weight_total:
          continue
        for i in range(len(action_rewards[act][rew][0])):
          reward_prob += action_rewards[act][rew][1][i] * (action_rewards[act][rew][0][i] / weight_total)
        weighted_act_rew[act] += reward_prob * float(rew.split("=",1)[1])
    optimal = dict_from_hash(max_key(self.rng, weighted_act_rew))
    return optimal if optimal else self.choose_random()

  def thompson_sample(self, givens):
    div_nodes = {a: self.div_nodes(a) for a in self.databank}
    CPTs = {}
    for node in self.environment.get_non_act_vars():
      CPTs[node] = DataSet()
      for agent, data in self.databank.items():
        if node not in div_nodes[agent]:
          CPTs[node].extend(data)
    max_sample = 0
    choice = None
    for action in permutations(self.action_domain):
      alpha_beta = self.get_alpha_beta(CPTs, div_nodes, action, givens)
      sample = self.rng.beta(alpha_beta[0], alpha_beta[1])
      if sample > max_sample:
        max_sample = sample
        choice = action
    return choice

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



  def get_alpha_beta(self, CPTs, div_nodes, action, givens):
    sol = 0
    for w in (0,1):
      pt1 = Query({"Y": 1}, {**{"W": w}, **givens}).solve(CPTs["Y"])
      pt2 = Query({"W": w}, action).solve(CPTs["W"])
      if pt1 is None or pt2 is None:
        return (1,1)
      sol += pt1 * pt2
    weight = len(CPTs["Y"].query(givens)) + len(CPTs["W"].query({**givens, **action}))
    alpha = 1 + sol * weight
    beta = 1 + (1-sol) * weight
    return (alpha, beta)






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
        
  
  # def thompson_sample(self, givens):
  #   choice = None
  #   max_sample = 0 #float('-inf')
  #   for action in permutations(self.action_domain):
  #     alpha, beta = 1, 1
  #     for agent in self.databank:
  #       transport_formula = self.transport_formula(agent, givens)
  #       transport_formula.assign(action)
  #       num_datapoints = self.get_num_datapoints(transport_formula, agent)
  #       if not num_datapoints:
  #         continue
  #       transport_formula.assign({self.reward_var: 1})
  #       a = self.solve_transport_formula(transport_formula, agent)
  #       a = 0 if a is None else a * num_datapoints
  #       b = num_datapoints - a
  #       alpha += a
  #       beta += b
  #     sample = self.rng.beta(alpha, beta)
  #     if sample > max_sample:
  #       choice = action
  #       max_sample = sample
  #   return choice #if choice else self.choose_random()
