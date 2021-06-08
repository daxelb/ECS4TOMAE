import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from assignment_models import AssignmentModel, discrete_model, random_model, action_model
import numpy as np
import random
import util
from enums import Policy

class World:
  def __init__(self, agents):
    self.agents = agents
    self.cdn = self.get_correct_div_nodes()
    self.episodes = []
    
    self.total_correct = []
    return
  
  def update_episode_data(self):
    new_data = {}
    new_data["%_correct"] = self.get_perc_correct()
    self.episodes.append(new_data)

  def run(self, episodes=250):
    for _ in range(episodes):
      self.act()
      self.encounter_all()
      self.update_episode_data()
    self.plot_agent_perc_corr_div_ids()

  def plot_agent_perc_corr_div_ids(self):
    for a in self.agents:
      if a.policy == Policy.DEAF: continue
      plt.plot(
        np.arange(len(self.episodes)),
        np.array(util.list_from_list_of_dicts(self.episodes, "%_correct", a.name)),
        label=a.name
      )
    plt.xlabel("Iterations")
    plt.ylabel("% Correct IDs of S-node locations")
    plt.legend()
    plt.show()
    
  def plot_total_perc_corr_div_ids(self):
    plt.plot(
        np.arange(len(self.episodes)),
        np.array(util.list_from_list_of_dicts(self.episodes, "%_correct", "total")),
        label="total"
    )
    plt.xlabel("Iterations")
    plt.ylabel("% Correct IDs of S-node locations")
    plt.show()

  def act(self):
    [a.act() for a in self.agents]

  def encounter_all(self):
    for a in self.agents:
      friends = [f for f in self.agents if f != a]
      if not friends: return False
      [a.encounter(f) for f in friends]
    
  def encounter_one(self):
    for a in self.agents:
      friends = [f for f in self.agents if f != a]
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
    Agent("zero", Environment(baseline), "Y", policy=Policy.NAIVE),
    Agent("one", Environment(baseline), "Y"),
    Agent("two", Environment(w1), "Y"),
    Agent("three", Environment(w9), "Y"),
    Agent("four", Environment(z5), "Y")
  ]
  world = World(agents)
  world.run(50)