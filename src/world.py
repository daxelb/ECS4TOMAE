from agent import SensitiveAgent, AdjustAgent
import gutil
from copy import copy


class World:
  def __init__(self, agents, is_community=False):
    self.agents = agents
    self.is_community = is_community
    self.pseudo_cum_regret = {a: [0] for a in self.agents}
    self.has_sensitive = any([isinstance(a, (SensitiveAgent, AdjustAgent)) for a in self.agents])
    self.databank = agents[0].databank

  def run_once(self): 
    [a.act() for a in self.agents]
    if self.has_sensitive:
      self.databank.update_divergence()
    self.update_pseudo_regret()
    return
  
  def update_pseudo_regret(self):
    for a in self.agents:
      recent = a.get_recent()
      rew_received = recent[a.reward_var]
      rew_optimal = a.environment.optimal_reward(gutil.only_given_keys(recent, a.environment.feat_vars))
      curr_regret = self.pseudo_cum_regret[a][-1]
      new_regret = curr_regret + (rew_optimal - rew_received)
      self.pseudo_cum_regret[a].append(new_regret)
  
  def __copy__(self):
    return World(copy(self.agents), self.is_community)
  
  def __eq__(self, other):
    return self.__class__ == other.__class__ and \
           self.agents == other.agents and \
           self.databank == other.databank
