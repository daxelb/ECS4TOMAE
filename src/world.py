import numpy as np
import random
from agent import SensitiveAgent, AdjustAgent
import gutil
from copy import copy


class World:
  def __init__(self, agents):
    self.agents = agents
    self.pseudo_cum_regret = []
    self.hasSensitive = False
    self.databank = agents[0].databank
    for agent in self.agents:
      if isinstance(agent, (SensitiveAgent, AdjustAgent)):
        self.hasSensitive = True
        break

  def run_once(self): 
    self.act()
    if self.hasSensitive:
      self.databank.update_divergence()
    self.update_pseudo_cum_regret()
    return

  def run(self, episodes=250):
    for i in range(episodes):
      self.run_once()
      gutil.printProgressBar(i+1, episodes)
    return

  def act(self):
    [a.act() for a in self.agents]
    return

  def update_community_pseudo_regret(self):
    cum_regret = self.pseudo_cum_regret[-1] if self.pseudo_cum_regret else 0
    for a in self.agents:
      recent = a.get_recent()
      rew_received = recent[a.reward_var]
      rew_optimal = a.environment.optimal_reward(gutil.only_given_keys(recent, a.environment.feat_vars))
      cum_regret += (rew_optimal - rew_received)
    self.pseudo_cum_regret.append(cum_regret)
  
  def update_pseudo_cum_regret(self):
    """
    Returns a dict of the cumulative 
    regret of each agent (and total cum. regret)
    at this episode.
    """
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
    return World(copy(self.agents))
