import matplotlib.pyplot as plt
import numpy as np
import random
import gutil
from enums import Policy
from copy import copy

class World:
  def __init__(self, agents):
    self.agents = agents
    self.episodes = []
    self.hasSensitive = False
    for agent in self.agents:
      if agent.policy in [Policy.SENSITIVE, Policy.ADJUST]:
        self.hasSensitive = True
        break

  
  def update_episode_data(self):
    self.episodes.append(self.get_cum_regret())

  def run_once(self):
    self.act()
    if self.hasSensitive:
      self.update_divergence()
    self.update_episode_data()
    return

  def run(self, episodes=250):
    for i in range(episodes):
      self.run_once()
      gutil.printProgressBar(i+1, episodes)
    return

  def act(self):
    [a.act() for a in self.agents]
    return

  def update_divergence(self):
    self.agents[0].knowledge.databank.update_divergence()

  def encounter_all(self):
    """
    An encountering policy in which the agent
    encounters (and potentially communicates with)
    all agents in the world.
    """
    for a in self.agents:
      [a.encounter(f) for f in self.agents if a != f]
    return
    
  def encounter_one(self):
    """
    An encountering policy where agents
    encounter (and potentially communicate w/)
    a single agent, chosen at random, rather than
    (above) all agents.
    """
    for a in self.agents:
      a.encounter(random.choice([f for f in self.agents if f != a]))
    return
  
  def get_cum_regret(self):
    """
    Returns a dict of the cumulative 
    regret of each agent (and total cum. regret)
    at this episode.
    """
    cum_regret = gutil.Counter()
    for a in self.agents:
      recent = a.get_recent()
      rew_received = recent[a.rew_var]
      rew_optimal = a.environment.optimal_reward(gutil.only_given_keys(recent, a.environment.feat_vars))
      curr_regret = self.episodes[-1][a] if self.episodes else 0
      new_regret = curr_regret + (rew_optimal - rew_received)
      cum_regret[a] = new_regret
    return cum_regret
  
  def __copy__(self):
    return World(copy(self.agents))