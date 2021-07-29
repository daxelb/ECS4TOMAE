from knowledge import KnowledgeAdjust, Knowledge, KnowledgeNaive, KnowledgeSensitive
import util
import gutil
import random
from enums import Policy
class Agent:
  def __init__(self, hash, environment, databank, policy, epsilon=0.03, div_node_conf=0, samps_needed=0):
    self.hash = hash
    self.environment = environment
    self.act_vars = environment.get_act_vars()
    self.act_doms = environment.get_act_doms()
    self.rew_var = environment.get_rew_var()
    self.policy = policy
    self.epsilon = epsilon
    self.div_node_conf = div_node_conf
    self.samps_needed = samps_needed
    if policy == Policy.DEAF:
      self.knowledge = Knowledge(self, databank)
    elif policy == Policy.NAIVE:
      self.knowledge = KnowledgeNaive(self, databank)
    elif policy == Policy.SENSITIVE:
      self.knowledge = KnowledgeSensitive(self, databank, div_node_conf, samps_needed)
    elif policy == Policy.ADJUST:
      self.knowledge = KnowledgeAdjust(self, databank, div_node_conf, samps_needed)

  def get_recent(self):
    return self.knowledge.get_recent()

  def act(self):
    givens = self.environment.pre.sample()
    choice = self.choose(givens)
    givens |= choice
    self.knowledge.add_sample(self.environment.post.sample(givens))
    # self.databank.append(self, self.environment.post.sample(givens))
    # self.knowledge.add_sample(self, self.environment.post.sample(givens))

  def choose(self, givens={}):
    optimal_choice = self.knowledge.optimal_choice(givens)
    if random.random() < self.epsilon or not optimal_choice:
      return self.experiment(givens)
    return optimal_choice

  def experiment(self, givens={}):
    for choice in gutil.permutations(self.act_doms):
      if len(self.knowledge.my_data().query({**choice, **givens})) == 0:
        return choice
    return self.random_action()
    # reward_vals = util.reward_vals(
    #   self.knowledge.my_data(), self.act_vars, self.rew_var, givens)
    # unexplored = [util.dict_from_hash(e) for e in util.hashes_from_domain(self.act_doms) if e not in reward_vals.keys()]
    # return random.choice(unexplored) if unexplored else self.random_action()

  def random_action(self):
    return random.choice(gutil.permutations(self.act_doms))

  # def encounter(self, other):
  #   if self.policy == Policy.DEAF:
  #     return
  #   self.knowledge.add_sample(other, other.get_recent())
    
  def __copy__(self):
    return Agent(self.hash, self.environment, self.knowlege.databank, self.policy, self.epsilon, self.div_node_conf, self.samps_needed)

  def __hash__(self):
    return self.hash
  
  def __reduce__(self):
    return (self.__class__, (self.hash, self.environment, self.knowledge.databank, self.policy, self.epsilon, self.div_node_conf, self.samps_needed))

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
        and self.hash == other.hash \
        and self.environment == other.environment
        
  def __str__(self):
    return "Agent" + str(self.hash)
  
  def __repr__(self):
    return "<Agent" + self.hash + ": " + self.policy.value + ">"
