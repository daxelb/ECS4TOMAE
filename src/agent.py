from knowledge import KnowledgeAdjust, KnowledgeDeaf, KnowledgeNaive, KnowledgeSensitive
import util
import gutil
import random
from enums import Datatype, Policy

# DIV_NODE_CONF = 0.09
# SAMPS_NEEDED = 15
# DIV_EPS_DEC_SLOWNESS = 1.75
DIV_NODE_CONF = 0.075
SAMPS_NEEDED = 12
DIV_EPS_DEC_SLOWNESS = 5

class Agent:
  def __init__(self, name, environment, epsilon=0.05, policy=Policy.DEAF):
    self.name = name
    self.environment = environment
    self.epsilon = epsilon
    self.policy = policy
    self.rew_var = self.environment.rew_var
    self.rew_dom = gutil.only_given_keys(self.environment.domains, [self.rew_var])
    self.friends = {}
    self.act_vars = self.environment.act_vars
    
    if policy == Policy.DEAF:
      self.knowledge = KnowledgeDeaf(self.environment, self)
    elif policy == Policy.NAIVE:
      self.knowledge = KnowledgeNaive(self.environment, self)
    elif policy == Policy.SENSITIVE:
      self.knowledge = KnowledgeSensitive(self.environment, self)
    elif policy == Policy.ADJUST:
      self.knowledge = KnowledgeAdjust(self.environment, self)
  
    self.act_doms = gutil.only_given_keys(self.environment.domains, self.act_vars)

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
    self.knowledge.add_sample(other, other.get_recent())

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and self.name == other.name \
        and self.environment == other.environment \
        and self.rew_var == other.rew_var
