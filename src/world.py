from agent import SensitiveAgent, AdjustAgent
import gutil
from copy import copy


class World:
  def __init__(self, agents, is_community=False):
    self.agents = agents
    self.is_community = is_community
    self.pseudo_cum_regret = []
    self.has_sensitive = any([isinstance(a, (SensitiveAgent, AdjustAgent)) for a in self.agents])
    self.databank = agents[0].databank

  def run_once(self): 
    self.act()
    if self.has_sensitive:
      self.databank.update_divergence()
    if self.is_community:
      self.update_community_pseudo_regret()
    else:
      self.update_agent_pseudo_regret()
    return

  def run(self, episodes=250):
    for i in range(episodes):
      self.run_once()
      gutil.printProgressBar(i+1, episodes)
    return

  def act(self):
    [a.act() for a in self.agents]
    return

  def update_community_pseudo_regret(self, ind_var):
    cum_regret = self.pseudo_cum_regret[-1][ind_var] if self.pseudo_cum_regret else 0
    for a in self.agents:
      recent = a.get_recent()
      rew_received = recent[a.reward_var]
      rew_optimal = a.environment.optimal_reward(gutil.only_given_keys(recent, a.environment.feat_vars))
      cum_regret += (rew_optimal - rew_received)
    self.pseudo_cum_regret.append({ind_var: cum_regret})
  
  def update_agent_pseudo_regret(self, ind_var):
    cum_regret = gutil.Counter()
    for a in self.agents:
      recent = a.get_recent()
      rew_received = recent[a.reward_var]
      rew_optimal = a.environment.optimal_reward(gutil.only_given_keys(recent, a.environment.feat_vars))
      curr_regret = self.pseudo_cum_regret[-1][a] if self.pseudo_cum_regret else 0
      new_regret = curr_regret + (rew_optimal - rew_received)
      cum_regret[a] = new_regret
    self.pseudo_cum_regret.append(cum_regret)
  
  def __copy__(self):
    return World(copy(self.agents), self.is_community)
