from copy import copy, deepcopy
from cpt import CPT
from query import Count, Product, Query, Summation
from util import only_given_keys, permutations, only_dicts_with_givens, hellinger_dist
from enums import ASR
from math import inf


class Agent:
  def __init__(self, rng, name, environment, agents, tau=None, asr=ASR.EG, epsilon=0, rand_trials=0, cooling_rate=0):
    self.rng = rng
    self.name = name
    self._environment = environment
    self.cgm = environment.cgm
    self.agents = agents
    self.domains = environment.domains
    self.act_var = environment.act_var
    self.act_dom = self.domains[self.act_var]
    self.actions = permutations(only_given_keys(self.domains, [self.act_var]))
    self.rew_var = environment.rew_var
    self.rew_dom = self.domains[self.rew_var]
    self.rewards = permutations(only_given_keys(self.domains, [self.rew_var]))
    self.contexts = permutations(self.get_context())
    self.tau = tau
    self.asr = asr
    self.epsilon = [1] * len(self.contexts.keys()) if asr == ASR.ED else epsilon
    self.rand_trials = rand_trials
    self.rand_trials_rem = [rand_trials] * len(self.contexts)
    self.cooling_rate = cooling_rate
    self.my_cpts = {
        var: CPT(var, self.cgm.get_parents(var), self.domains)
        for var in self.domains
    }
    # nodes in the cgm that Y is dependent on that is either X or observed by X
    # in the OG example, this is Z, X
    # but if Z is not a counfounder on Y, but only connected to Y through X,
    # Z would not be included
    parents = {self.act_var}
    for var in self.cgm.get_ancestors(self.act_var):
      if not self.cgm.is_d_separated(var, self.rew_var, self.act_var):
        parents.add(var)
    self.my_cpts["rew"] = CPT(self.rew_var, parents, self.domains)

  def update_divergence(self):
    return

  def get_context(self):
    return only_given_keys(self.domains, self.cgm.get_parents(self.act_var))

  def get_ind_var_value(self, ind_var):
    """
    Returns the assignment of the givent
    "indepedent variable."
    """
    if ind_var == "tau":
      return self.tau
    elif ind_var == "otp":
      return self.get_otp()
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

  def choose(self, givens):
    """
    Defines the logic of how the agent
    'chooses' according their Action 
    Selection Rule (ASR).
    """
    if self.asr == ASR.EG:
      if self.rng.random() < self.epsilon:
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == ASR.EF:
      if self.feat_perms:
        given_i = self.feat_perms.index(givens)
        if self.rand_trials_rem[given_i] > 0:
          self.rand_trials_rem[given_i] -= 1
          return self.choose_random()
      elif self.rand_trials_rem > 0:
        self.rand_trials_rem -= 1
        return self.choose_random()
      return self.choose_optimal(givens)
    elif self.asr == ASR.ED:
      if self.feat_perms:
        given_i = self.feat_perms.index(givens)
        if self.rng.random() < self.epsilon[given_i]:
          self.epsilon[given_i] *= self.cooling_rate
          return self.choose_random()
      elif self.rng.random() < self.epsilon:
        self.epsilon *= self.cooling_rate
        return self.choose_random()
      self.epsilon *= self.cooling_rate
    elif self.asr == ASR.TS:
      return self.thompson_sample(givens)
    else:
      raise ValueError("%s ASR not found" % self.asr)

  def observe(self, sample):
    """
    The behavior of the agent returning 
    """
    self.recent = sample
    for cpt in self.my_cpts.values():
      cpt.add(sample)

  def get_recent(self):
    return self.recent

  def get_rew_query_unfactored(self):
    parents = {self.act_var}
    for var in self.cgm.get_ancestors(self.act_var):
      if not self.cgm.is_d_separated(var, self.rew_var, self.act_var):
        parents.add(var)
    return Query(self.rew_var, parents)

  def expected_rew(self, givens, cpts):
    summ = 0
    query = self.get_rew_query_unfactored()
    query.assign(givens)
    for rew in self.rewards:
      query.assign(rew)
      rew_prob = query.solve(cpts["rew"])
      summ += rew[self.rew_var] * rew_prob if rew_prob is not None else 0
    return summ

  def choose_optimal(self, context):
    cpts = self.get_cpts()
    best_acts = []
    best_rew = -inf
    for act in self.actions:
      expected_rew = self.expected_rew({**context, **act}, cpts)
      if expected_rew is not None:
        if expected_rew > best_rew:
          best_acts = [act]
          best_rew = expected_rew
        elif expected_rew == best_rew:
          best_acts.append(act)
    return self.rng.choice(best_acts) if best_acts else None

  def choose_random(self):
    """
    Randomly chooses from the possible acti
    """
    return self.rng.choice(self.actions)

  def thompson_sample(self, context):
    best_acts = []
    best_sample = 0
    cpts = self.get_cpts()
    rew_query = self.get_rew_query_unfactored()
    rew_query = Count(rew_query)
    rew_query.assign(context)
    for act in self.actions:
      rew_query.assign(act)
      alpha = cpts["rew"][rew_query.assign({self.rew_var: 1})]
      beta = cpts["rew"][rew_query.assign({self.rew_var: 0})]
      sample = self.rng.beta(alpha + 1, beta + 1)
      if sample > best_sample:
        best_sample = sample
        best_acts = [act]
      if sample == best_sample:
        best_acts.append(act)
    return self.rng.choice(best_acts)

  def get_otp(self):
    return self.__class__.__name__[:-5]

  def __hash__(self):
    return hash(self.name)

  def __reduce__(self):
    return (self.__class__, (self.rng, self.name, self.environment, self.databank, self.tau, self.asr, self.epsilon, self.rand_trials, self.cooling_rate))

  def __repr__(self):
    return "<" + self.get_otp() + self.name + ": " + self.asr.value + ">"

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and self.name == other.name


class SoloAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_cpts(self):
    return self.my_cpts


class NaiveAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_cpts(self):
    cpts = copy(self.my_cpts)
    for n in cpts:
      for a in self.agents:
        if a == self:
          continue
        cpts[n].update(a.my_cpts[n])
    return cpts


class SensitiveAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # May want to include all ancestors of X, not just parents
    self.context_vars = set(self.get_context().keys())
    self.divergence = dict()

  def get_cpts(self):
    # might not be deepcopying here...
    cpts = copy(self.my_cpts)
    for a in self.agents:
      if a == self:
        continue
      if self.div_nodes(a).issubset(self.context_vars):
        (cpts[n].update(a.my_cpts[n]) for n in cpts)
    return cpts

  def get_non_act_nodes(self):
    return {node for node in self.domains if node != self.act_var}

  def update_divergence(self):
    for a in self.agents:
      if a == self:
        continue
      if a not in self.divergence:
        self.divergence[a] = {n: inf for n in self.cgm.get_unset_nodes()}
      for n in self.get_non_act_nodes():
        self.divergence[a][n] = hellinger_dist(
            self.domains, self.my_cpts[n], a.my_cpts[n], self.cgm.get_node_dist(n))

  def div_nodes(self, agent):
    if self == agent:
      return set()
    return {node for node, dist in self.divergence[agent].items() if dist is None or dist > self.get_scaled_tau(node)}

  def get_scaled_tau(self, node):
    scale_factor = 1
    for parent in self.cgm.get_parents(node):
      scale_factor *= len(self.domains[parent])
    return self.tau * scale_factor


class AdjustAgent(SensitiveAgent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_cpts(self):
    cpts = deepcopy(self.my_cpts)
    for a in self.agents:
      if a == self:
        continue
      div_nodes = self.div_nodes(a)
      for n in cpts:
        if n not in div_nodes:
          cpts[n].update(a.my_cpts[n])
    return cpts

  def expected_rew(self, givens, cpts):
    query = self.get_rew_query().assign(givens)
    summ = 0
    for rew_val in self.rew_dom:
      query.assign(self.rew_var, rew_val)
      rew_prob = query.solve(cpts)
      summ += rew_val * rew_prob if rew_prob is not None else 0
    return summ

  def get_rew_query(self):
    dist_vars = self.cgm.causal_path(self.act_var, self.rew_var)
    return Product(
        self.cgm.get_node_dist(v)
        for v in dist_vars
    ).assign(self.domains)

  def all_causal_path_nodes_corrupted(self, agent):
    return self.cgm.causal_path(self.act_var, self.rew_var).issubset(set(self.div_nodes(agent)))

  def thompson_sample(self, context):
    best_acts = []
    best_sample = 0
    cpts = self.get_cpts()
    for act in self.actions:
      a = 0
      b = 0
      for agent in self.agents:
        if self.all_causal_path_nodes_corrupted(agent):
          continue
        rew_query = self.get_rew_query().assign(act).assign(context)
        a_prob = rew_query.assign(self.rew_var, 1).solve(cpts)
        b_prob = rew_query.assign(self.rew_var, 0).solve(cpts)
        if a_prob is None or b_prob is None:
          continue
        count = agent.my_cpts[self.act_var][Count({**act, **context})]
        a += a_prob * count
        b += b_prob * count
      sample = self.rng.beta(a+1, b+1)
      if sample > best_sample:
        best_sample = sample
        best_acts = [act]
      if sample == best_sample:
        best_acts.append(act)
    return self.rng.choice(best_acts)
