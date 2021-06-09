import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from assignment_models import AssignmentModel, discrete_model, random_model
import numpy as np
import random
import util
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
    for _ in range(episodes):
      self.act()
      self.encounter_all()
      self.update_episode_data()
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
      correct_div_nodes[a.name] = {}
      for f in self.agents:
        if a == f: continue
        correct_div_nodes[a.name][f.name] = {}
        for node, assignment in a.environment._assignment.items():
          correct_div_nodes[a.name][f.name][node] = assignment != f.environment._assignment[node]
    return correct_div_nodes

  def get_perc_correct(self):
    percent_correct = {}
    total_correct = 0
    for a in self.agents:
      if a.policy == Policy.DEAF: continue
      for f in self.agents:
        if a == f: continue
        correct = util.num_matches(
          a.divergent_nodes()[f.name],
          self.cdn[a.name][f.name]
        )
        perc_correct = correct / (len(self.cdn) - 1)
        total_correct += correct
        percent_correct[a.name] = perc_correct
    percent_correct["total"] = total_correct / ((len(self.cdn) - 1) * len(self.cdn) * len(util.first_value(util.first_value(self.cdn))))
    return percent_correct
  
  def get_cum_regret(self):
    cum_regret = {}
    cum_regret["total"] = 0
    for a in self.agents:
      recent = a.recent
      rew_received = recent[a.reward_var]
      rew_optimal = a.environment.optimal_reward(util.only_specified_keys(recent, a.environment.feature_nodes))
      curr_regret = self.episodes[-1][Result.CUM_REGRET][a.name] if self.episodes else 0
      new_regret = curr_regret + (rew_optimal - rew_received)
      cum_regret[a.name] = new_regret
      cum_regret["total"] += new_regret
    return cum_regret

  def plot_agent(self, dep_var):
    for a in self.agents:
      if dep_var== Result.PERC_CORR and a.policy == Policy.DEAF: continue
      plt.plot(
        np.arange(len(self.episodes)),
        np.array(util.list_from_dicts(self.episodes, dep_var, a.name)),
        label=a.name
      )
    plt.xlabel("Iterations")
    plt.ylabel(str(dep_var))
    plt.legend()
    plt.show()
    return
    
  def plot_total(self, dep_var):
    if "total" not in self.episodes[dep_var]:
      raise KeyError("Total is not tracked in the input dep_var, {}".format(dep_var))
    plt.plot(
        np.arange(len(self.episodes)),
        np.array(util.list_from_dicts(self.episodes, dep_var, "total")),
        label="total"
    )
    plt.xlabel("Iterations")
    plt.ylabel(str(dep_var))
    plt.show()
    return
  
  def plot_policy(self, dep_var):
    policies = {}
    for a in self.agents:
      if a.policy not in policies:
          policies[a.policy] = []
      policies[a.policy].append(util.list_from_dicts(self.episodes, dep_var, a.name))
    for p in policies:
      plt.plot(
        np.arange(len(self.episodes)),
        np.array(util.avg_list(policies[p])),
        label=p
      )
    plt.xlabel("Iterations")
    plt.ylabel(str(dep_var))
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
  baseline = {
    "W": random_model((0.5, 0.5)),
    "X": AssignmentModel(("W"), None, (0, 1)),
    "Z": discrete_model(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": discrete_model(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  }
  w1 = dict(baseline)
  w1["W"] = random_model((0.1, 0.9))
  w9 = dict(baseline)
  w9["W"] = random_model((0.9, 0.1))
  z5 = dict(baseline)
  z5["Z"] = discrete_model(("X"), {(0,): (0.8, 0.2), (1,): (0.5, 0.5)})

  agents = [
    Agent("zero", Environment(baseline), policy=Policy.NAIVE),
    Agent("one", Environment(baseline), policy=Policy.SENSITIVE),
    Agent("two", Environment(baseline)),
    Agent("three", Environment(w1)),
    Agent("four", Environment(w1)),
    Agent("five", Environment(w1)) 
  ]
  world = World(agents)
  world.run(50)
  world.plot_policy(Result.CUM_REGRET)