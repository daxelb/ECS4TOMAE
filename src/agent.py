from knowledge import KnowledgeAdjust, Knowledge, KnowledgeNaive, KnowledgeSensitive
import util
import gutil
import random
from enums import Policy
class Agent:
  def __init__(self, name, environment, policy, epsilon=0.03, div_node_conf=0, samps_needed=0):
    self.name = name
    self.environment = environment
    self.policy = policy
    self.epsilon = epsilon
    self.div_node_conf = div_node_conf
    self.samps_needed = samps_needed
    self.rew_var = self.environment.rew_var
    self.rew_dom = gutil.only_given_keys(self.environment.domains, [self.rew_var])
    self.act_vars = self.environment.act_vars
    self.act_doms = gutil.only_given_keys(self.environment.domains, self.act_vars)
    if policy == Policy.DEAF:
      self.knowledge = Knowledge(self)
    elif policy == Policy.NAIVE:
      self.knowledge = KnowledgeNaive(self)
    elif policy == Policy.SENSITIVE:
      self.knowledge = KnowledgeSensitive(self, div_node_conf, samps_needed)
    elif policy == Policy.ADJUST:
      self.knowledge = KnowledgeAdjust(self, div_node_conf, samps_needed)

  def get_recent(self):
    return self.knowledge.my_data()[-1]

  def act(self):
    givens = self.environment.pre.sample()
    choice = self.choose(givens)
    givens |= choice
    self.knowledge.add_sample(self, self.environment.post.sample(givens))

  def choose(self, givens={}):
    optimal_choice = self.knowledge.optimal_choice(givens)
    if random.random() < self.epsilon or not optimal_choice:
      return self.experiment(givens)
    return optimal_choice

  def experiment(self, givens={}):
    reward_vals = util.reward_vals(
      self.knowledge.my_data(), self.act_vars, self.rew_var, givens)
    unexplored = [util.dict_from_hash(e) for e in util.hashes_from_domain(self.act_doms) if e not in reward_vals.keys()]
    return random.choice(unexplored) if unexplored else self.random_action()

  def random_action(self):
    return random.choice(gutil.permutations(self.act_doms))

  def encounter(self, other):
    if self.policy == Policy.DEAF:
      return
    self.knowledge.add_sample(other, other.get_recent())
    
  def __copy__(self):
    print("!!")
    return Agent(self.name, self.environment, self.policy, self.epsilon, self.div_node_conf, self.samps_needed)

  def __hash__(self):
    return hash(self.name)
  
  def __reduce__(self):
    return (self.__class__, (self.name, self.environment, self.policy, self.epsilon, self.div_node_conf, self.samps_needed))

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and self.name == other.name \
        and self.environment == other.environment \
        and self.rew_var == other.rew_var
