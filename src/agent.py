from knowledge import Knowledge
import util
import gutil
import random
from enums import Datatype, Policy

# DIV_NODE_CONF = 0.09
# SAMPS_NEEDED = 15
# DIV_EPS_DEC_SLOWNESS = 1.75
DIV_NODE_CONF = 0.06
SAMPS_NEEDED = 10
DIV_EPS_DEC_SLOWNESS = 2.5

class Agent:
  def __init__(self, name, environment, epsilon=0.01, policy=Policy.DEAF):
    self.name = name
    self.environment = environment
    self.epsilon = epsilon
    self.policy = policy
    self.rew_var = self.environment.rew_var
    self.rew_dom = gutil.only_given_keys(self.environment.domains, [self.rew_var])
    self.friends = {}
    self.act_vars = self.environment.act_vars
    self.knowledge = Knowledge(self.environment)
    self.act_doms = gutil.only_given_keys(self.environment.domains, self.act_vars)
    self.friend_divergence = {}

  def act(self):
    givens = self.environment.pre.sample()
    choice = self.choose(givens)
    givens |= choice[1]
    # env_act_feedback = self.environment.post.sample(givens)
    self.knowledge.add_sample(self.environment.post.sample(givens))
    # self.knowledge.add_obs(env_act_feedback) \
    #     if choice[0] == Datatype.OBS \
    #     else self.knowledge.add_exp(env_act_feedback)

  def choose(self, givens={}):
    if random.random() < self.epsilon:
      return (Datatype.EXP, self.experiment(givens))
    else:
      optimal_choice = self.optimal_choice(givens)
      return (Datatype.OBS, optimal_choice) if optimal_choice else (Datatype.EXP, self.experiment(givens))

  def experiment(self, givens={}):
    reward_vals = util.reward_vals(
      self.knowledge.get_useful_data(), self.act_vars, self.rew_var, givens)
    unexplored = [util.dict_from_hash(e) for e in util.hashes_from_domain(self.act_doms) if e not in reward_vals.keys()]
    return random.choice(unexplored) if unexplored\
      else self.random_action()

  def optimal_choice(self, givens={}):
    my_data = self.get_data().copy()
    
    if self.policy == Policy.NAIVE:
      for f in self.friends:
        my_data.extend(self.friends[f])
    
    if self.policy == Policy.SENSITIVE:
      for f in self.friends:
        if True not in self.friend_divergence[f].values():
          my_data.extend(self.friends[f])
          
    if self.policy == Policy.ADJUST:
     return self.optimal_choice_adjustment(givens)
    
    expected_values = util.expected_vals(
        my_data, self.act_vars, self.rew_var, givens)
    return util.dict_from_hash(gutil.max_key(expected_values)) if expected_values\
      else self.random_action()
  
  def optimal_choice_adjustment(self, givens={}):
    dataset = {**self.friends, **{self.name: self.get_data()}}
    
    action_rewards = {}
    for action in gutil.permutations(self.act_doms):
      act_hash = util.hash_from_dict(action)
      action_rewards[act_hash] = {}
      for rew in gutil.permutations(self.rew_dom):
        action_rewards[act_hash][util.hash_from_dict(rew)] = [[],[]]
    
    for agent, data in dataset.items():
      transport_formula = self.transport_formula(agent, givens)
      if not transport_formula:
        continue
      S_domains = gutil.only_given_keys(
          self.knowledge.domains,
          [key for key in transport_formula[1][0]]
      ) if len(transport_formula) > 1 else False
      for action in gutil.permutations(self.act_doms):
        act_hash = util.hash_from_dict(action)
        num_datapoints = len(gutil.only_dicts_with_givens(data, {**action, **givens}))
        for rew in gutil.permutations(self.rew_dom):
          rew_hash = util.hash_from_dict(rew)
          assignments = []
          if S_domains:
            for s in gutil.permutations(S_domains):
              assignments.append({**rew, **action, **givens, **s})
          else:
            assignments.append({**rew, **action, **givens})
          summation = 0
          for assignment in assignments:
            if not gutil.only_dicts_with_givens(data, assignment):
              continue
            product = 1
            for factor in util.apply_assignments_to_queries(transport_formula, assignment):
              product *= util.prob(data, factor[0], factor[1])
            summation += product
          action_rewards[act_hash][rew_hash][0].append(num_datapoints)
          action_rewards[act_hash][rew_hash][1].append(summation)
    
    weighted_act_rew = gutil.Counter()
    for act in action_rewards:
      for rew in action_rewards[act]:
        reward_prob = 0
        weight_total = sum(action_rewards[act][rew][0])
        if not weight_total: continue
        for i in range(len(action_rewards[act][rew][0])):
          weight = action_rewards[act][rew][0][i] / weight_total
          reward_prob += action_rewards[act][rew][1][i] * weight
        weighted_act_rew[act] += reward_prob * float(rew.split("=",1)[1])
    # print(weighted_act_rew)
    return util.dict_from_hash(gutil.max_key(weighted_act_rew)) if weighted_act_rew else self.random_action()
        
    # my_expected_vals = util.expected_vals(self.get_data(), self.act_vars, self.rew_var, givens)
    # # print(my_expected_vals)
    # action_rewards = {}
    # for action in gutil.permutations(self.act_doms):
    #   action_hash = util.hash_from_dict(action)
    #   if action_hash not in my_expected_vals:
    #     return None
    #   action_rewards[action_hash] = {"exp_rew": [my_expected_vals[action_hash]], "weights": [len(self.get_data())]}
    # for f in self.friends:
    #   # transport formula currently only supports one choice node, so if
    #   # there are multiple choice nodes in the future, these methods must be updated
    #   transport_formula = self.transport_formula(f, givens)
    #   if transport_formula is None: continue
    #   if len(transport_formula) > 1:
    #     S_keys = [key for key in transport_formula[1][0]] 
    #     S_domains = gutil.only_given_keys(self.knowledge.domains, S_keys)
    #   # print(self.name, f, transport_formula)
    #   for action in gutil.permutations(self.act_doms):
    #     exp_rew_of_action = 0
    #     for rew in self.knowledge.domains[self.rew_var]:
    #       if len(transport_formula) > 1:
    #         summation_over_S = 0
    #         for S_assignments in gutil.permutations(S_domains):
    #           assignments = {**action, **{self.rew_var: rew}, **S_assignments, **givens}
    #           query_factors = util.apply_assignments_to_queries(transport_formula, assignments)
    #           product = 1
    #           for factor in query_factors:
    #             if not gutil.only_dicts_with_givens(self.friends[f], factor[1]):
    #               break
    #             product *= util.prob(self.friends[f], factor[0], factor[1])
    #           summation_over_S += product
    #       else:
    #         assignments = {**action, **{self.rew_var: rew}, **givens}
    #         product = 1
    #         query_factors = util.apply_assignments_to_queries(transport_formula, assignments)
    #         for factor in query_factors:
    #           if not gutil.only_dicts_with_givens(self.friends[f], factor[1]):
    #               continue
    #           product *= util.prob(self.friends[f], factor[0], factor[1])
    #         summation_over_S = product
    #       exp_rew_of_action += rew * summation_over_S
    #     action_hash = util.hash_from_dict(action)
    #     action_rewards[action_hash]["exp_rew"].append(exp_rew_of_action)
    #     action_rewards[action_hash]["weights"].append(len(self.friends[f]))
      
    # exp_vals = {}
    # for action in action_rewards:
    #   exp_vals[action] = 0
    #   for i in range(len(action_rewards[action]["weights"])):
    #     exp_vals[action] += \
    #         (action_rewards[action]["weights"][i] / sum(action_rewards[action]["weights"])) \
    #         * action_rewards[action]["exp_rew"][i]
            
    # return util.dict_from_hash(gutil.max_key(exp_vals))
    
  # def query_product(self, query_factors, )

  def random_action(self):
    return random.choice(gutil.permutations(self.act_doms))

  def add_friend(self, other):
    self.friends[other.name] = []
    self.friend_divergence[other.name] = {}
    for node in self.knowledge.model.get_observable():
      if node not in self.act_vars:
        self.friend_divergence[other.name][node] = True

  def encounter(self, other):
    if self.policy == Policy.DEAF: return
    # friend_data = other.recent#other.knowledge.get_useful_data()
    if other.name not in self.friends:
      self.add_friend(other)
    other_recent = other.get_data()[-1]
    if other_recent:
      self.friends[other.name].append(other_recent)
    if self.policy == Policy.SENSITIVE or self.policy == Policy.ADJUST:
      self.update_friend_divergence()
    
  def get_data(self):
    return self.knowledge.samples

  def update_friend_divergence(self):
    for f in self.friends:
      friend_data = self.friends[f]
      div_epsilon = (SAMPS_NEEDED * DIV_EPS_DEC_SLOWNESS)/(len(friend_data) - SAMPS_NEEDED + SAMPS_NEEDED * DIV_EPS_DEC_SLOWNESS)
      if div_epsilon > 1: continue
      for node in self.friend_divergence[f]:
        if self.friend_divergence[f][node] == False:
          if random.random() >= div_epsilon:
            continue
        node_div = self.knowledge.kl_divergence_of_node(node, friend_data)
        if node_div != None and node_div < DIV_NODE_CONF:
          self.friend_divergence[f][node] = False
        else:
          break
    return
  
  def div_nodes(self, agent_name):
    if agent_name == self.name:
      return []
    return [node for node, divergent in self.friend_divergence[agent_name].items() if divergent]
      
  def selection_diagram(self, agent_name):
    # print(self.name, agent_name, self.div_nodes(agent_name))
    return self.environment.selection_diagram(self.div_nodes(agent_name))
  
  def transport_formula(self, friend, givens):
    return self.selection_diagram(friend).get_transport_formula(self.act_vars[0], self.rew_var, list(givens))

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and self.name == other.name \
        and self.environment == other.environment \
        and self.rew_var == other.rew_var
