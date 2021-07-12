import matplotlib.pyplot as plt
import numpy as np
import random
import gutil
from enums import Policy, Result
from copy import copy, deepcopy

class World:
  def __init__(self, agents):
    self.agents = agents
    self.cdn = self.get_correct_div_nodes()
    self.episodes = []
  
  def update_episode_data(self):
    new_data = {}
    new_data[Result.PERC_CORR] = self.get_perc_correct()
    new_data[Result.CUM_REGRET] = self.get_cum_regret()
    self.episodes.append(new_data)

  def run_once(self):
    self.act()
    self.encounter_all()
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
    
  def get_correct_div_nodes(self):
    """
    Returns a 3-layer dictionary which
    identifies the correct location of divergent
    nodes between two environments.
    """
    correct_div_nodes = {}
    for a in self.agents:
      if a.policy in [Policy.DEAF, Policy.NAIVE]: continue
      correct_div_nodes[a.name] = {}
      for f in self.agents:
        if a == f: continue
        correct_div_nodes[a.name][f.name] = {}
        for node, assignment in a.environment._assignment.items():
          if node in a.act_vars: continue
          correct_div_nodes[a.name][f.name][node] = assignment != f.environment._assignment[node]
    return correct_div_nodes

  def get_perc_correct(self):
    """
    Returns a dict of the percentage of 
    divergent/not divergent nodes correctly
    identified by each agent (and the total).
    """
    percent_correct = gutil.Counter()
    for a in self.agents:
      if a.policy in [Policy.DEAF, Policy.NAIVE]: continue
      for f in self.agents:
        if a == f:
          continue
        correct = gutil.num_matches(a.knowledge.is_divergent_dict(f), self.cdn[a.name][f.name])
        perc_corr = correct / (len(self.cdn[a.name]) * len(self.cdn[a.name][f.name]))
        percent_correct[a.name] += perc_corr
        percent_correct["total"] += perc_corr/len(self.cdn)
    return percent_correct
  
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
      rew_optimal = a.environment.optimal_reward(gutil.only_given_keys(recent, a.environment.feature_nodes))
      curr_regret = self.episodes[-1][Result.CUM_REGRET][a.name] if self.episodes else 0
      new_regret = curr_regret + (rew_optimal - rew_received)
      cum_regret[a.name] = new_regret
      cum_regret["total"] += new_regret
    return cum_regret

  def plot_agent(self, dep_var):
    for a in self.agents:
      if dep_var == Result.PERC_CORR and a.policy in [Policy.DEAF, Policy.NAIVE] : continue
      plt.plot(
        np.arange(len(self.episodes)),
        np.array(gutil.list_from_dicts(self.episodes, dep_var, a.name)),
        label=a.name
      )
    plt.xlabel("Iterations")
    plt.ylabel(dep_var.value)
    plt.legend()
    plt.show()
    return
    
  def plot_total(self, dep_var):
    if "total" not in self.episodes[0][dep_var]:
      raise KeyError("Total is not tracked in the input dep_var, {}".format(dep_var))
    plt.plot(
        np.arange(len(self.episodes)),
        np.array(gutil.list_from_dicts(self.episodes, dep_var, "total")),
        label="total"
    )
    plt.xlabel("Iterations")
    plt.ylabel(dep_var.value)
    plt.show()
    return
  
  def plot_policy(self, dep_var):
    policies = {}
    for a in self.agents:
      if a.policy not in policies:
          policies[a.policy] = []
      policies[a.policy].append(gutil.list_from_dicts(self.episodes, dep_var, a.name))
    for p in policies:
      plt.plot(
        np.arange(len(self.episodes)),
        np.array(gutil.avg_list(policies[p])),
        label=p
      )
    plt.xlabel("Iterations")
    plt.ylabel(dep_var.value)
    plt.legend()
    plt.show()
    return
  
  def __copy__(self):
    return World(self.agents.copy())