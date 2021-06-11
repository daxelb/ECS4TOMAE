import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from assignment_models import AssignmentModel, discrete_model, random_model
import numpy as np
import random
import gutil
from enums import Policy, Result

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

  def run(self, episodes=250):
    for i in range(episodes):
      self.act()
      self.encounter_all()
      self.update_episode_data()
      gutil.printProgressBar(i+1, episodes)
    return

  def act(self):
    [a.act() for a in self.agents]
    return
  
  def get_friends(self, agent):
    return [f for f in self.agents if f != agent]

  def encounter_all(self):
    for a in self.agents:
      [a.encounter(f) for f in self.get_friends(a)]
    return
    
  def encounter_one(self):
    for a in self.agents:
      friends = self.get_friends(a)
      if not friends: return False
      a.encounter(random.choice(friends))
    return True
    
  def get_correct_div_nodes(self):
    correct_div_nodes = {}
    for a in self.agents:
      if a.policy in [Policy.DEAF, Policy.NAIVE]: continue
      correct_div_nodes[a.name] = {}
      for f in self.agents:
        if a == f: continue
        correct_div_nodes[a.name][f.name] = {}
        for node, assignment in a.environment._assignment.items():
          if node not in a.act_vars:
            correct_div_nodes[a.name][f.name][node] = assignment != f.environment._assignment[node]
    return correct_div_nodes

  def get_perc_correct(self):
    percent_correct = gutil.Counter()
    for a in self.agents:
      if a.policy in [Policy.DEAF, Policy.NAIVE]: continue
      for f in self.agents:
        if a == f: continue
        correct = gutil.num_matches(a.friend_divergence[f.name], self.cdn[a.name][f.name])
        perc_corr = correct / (len(self.cdn[a.name]) * len(self.cdn[a.name][f.name]))
        percent_correct[a.name] += perc_corr
        percent_correct["total"] += perc_corr/len(self.cdn)
    return percent_correct
  
  def get_cum_regret(self):
    cum_regret = gutil.Counter()
    for a in self.agents:
      recent = a.get_data()[-1]
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

if __name__ == "__main__":
  baseline = {
    "W": random_model((0.4, 0.6)),
    "X": AssignmentModel(("W"), None, (0, 1)),
    "Z": discrete_model(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": discrete_model(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  }
  w1 = dict(baseline)
  w1["W"] = random_model((0.1, 0.9))
  w9 = dict(baseline)
  w9["W"] = random_model((0.9, 0.1))
  z5 = dict(baseline)
  z5["Z"] = discrete_model(("X"), {(0,): (0.9, 0.1), (1,): (0.5, 0.5)})
  reversed_z = dict(baseline)
  reversed_z["Z"] = discrete_model(("X"), {(0,): (0.25, 0.75), (1,): (0.75, 0.25)})

  agents = [
    Agent("00", Environment(baseline), policy=Policy.DEAF),
    Agent("01", Environment(baseline), policy=Policy.DEAF),
    Agent("02", Environment(baseline), policy=Policy.NAIVE),
    Agent("03", Environment(baseline), policy=Policy.NAIVE),
    Agent("04", Environment(baseline), policy=Policy.SENSITIVE),
    Agent("05", Environment(baseline), policy=Policy.SENSITIVE),
    Agent("06", Environment(reversed_z), policy=Policy.DEAF),
    Agent("07", Environment(reversed_z), policy=Policy.DEAF),
    Agent("08", Environment(reversed_z), policy=Policy.NAIVE),
    Agent("09", Environment(reversed_z), policy=Policy.NAIVE),
    Agent("10", Environment(reversed_z), policy=Policy.SENSITIVE),
    Agent("11", Environment(reversed_z), policy=Policy.SENSITIVE),
    Agent("12", Environment(z5), policy=Policy.DEAF),
    Agent("13", Environment(z5), policy=Policy.NAIVE),
    Agent("14", Environment(z5), policy=Policy.SENSITIVE),
  ]
  world = World(agents)
  world.run(220)
  world.plot_agent(Result.PERC_CORR)
  world.plot_total(Result.PERC_CORR)
  world.plot_policy(Result.CUM_REGRET)
