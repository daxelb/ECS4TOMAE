from world import World
from assignment_models import AssignmentModel, discrete_model, random_model
from agent import Agent
from environment import Environment
from enums import Policy, Result
import gutil
import matplotlib.pyplot as plt
import numpy as np

class Sim:
  def __init__(self, world, num_episodes, num_trials):
    self.world = world
    self.num_episodes = num_episodes
    self.num_trials = num_trials
    self.trials = []
  
  def run(self):
    for i in range(self.num_trials):
      world = self.world.__copy__()
      for j in range(self.num_episodes):
        world.run_once()
        gutil.printProgressBar(i+((j+1)/self.num_episodes), self.num_trials)
      self.trials.append(world.episodes)
    return
  
  def plot_policy(self, dep_var, filename):
    policies = {}
    for trial_episode in self.trials:
      for a in self.world.agents:
        if a.policy not in policies:
            policies[a.policy] = []
        policies[a.policy].append(gutil.list_from_dicts(trial_episode, dep_var, a.name))
    for p in policies:
      plt.plot(
        np.arange(self.num_episodes),
        np.array(gutil.avg_list(policies[p])),
        label=p
      )
    plt.xlabel("Iterations")
    plt.ylabel(dep_var.value)
    plt.legend()
    plt.savefig(filename)
    # plt.show()
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
  reversed_y = dict(baseline)
  reversed_y["Y"] = discrete_model(("W", "Z"), {(0, 0): (0, 1), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (1, 0)})

  for pol in [Policy.DEAF, Policy.NAIVE, Policy.SENSITIVE, Policy.ADJUST]:
    agents = [
      Agent("00", Environment(baseline), policy=pol),
      Agent("01", Environment(w1), policy=pol),
      Agent("02", Environment(w9), policy=pol),
      Agent("03", Environment(z5), policy=pol),
      Agent("04", Environment(reversed_z), policy=pol),
      Agent("05", Environment(reversed_y), policy=pol),
    ]
    sim = Sim(World(agents), 240, 12)
    sim.run()
  sim.plot_policy(Result.CUM_REGRET, "../output/6agent-240-allEnvDiff.png")
