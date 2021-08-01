from data import DataBank
from world import World
from assignment_models import ActionModel, DiscreteModel, RandomModel
from agent import Agent
from environment import Environment
from enums import Policy, IV
import gutil
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import time
from copy import copy

def get_data_key(agent, ind_var):
  if ind_var == IV.POL:
    return agent.policy
  elif ind_var == IV.EPS:
    return agent.epsilon
  elif ind_var == IV.DNC:
    return agent.div_node_conf
  elif ind_var == IV.SN:
    return agent.samps_needed

class Sim:
  def __init__(self, world, num_episodes, num_trials):
    self.world = world
    self.num_episodes = num_episodes
    self.num_trials = num_trials
    self.trials = []
  
  def run(self):
    for i in range(self.num_trials):
      world = copy(self.world)
      for j in range(self.num_episodes):
        world.run_once()
        gutil.printProgressBar(
          i+((j+1)/self.num_episodes),
          self.num_trials,
          "{}:{}".format(i,j),
          )
      self.trials.append(world.pseudo_cum_regret)
    return

  def get_data(self):
    data = pd.DataFrame(columns=range(self.num_episodes))
    for trial in self.trials:
      for agent in self.world.agents:
        data.loc[len(data)] = gutil.list_from_dicts(trial, agent)
    return data
  
  def simulate(self, results, index):
    sim = copy(self)
    sim.run()
    results[index] = sim.get_data()

  def multithreaded_sim(self):
    num_threads = mp.cpu_count()
    jobs = []
    results = mp.Manager().list([None] * num_threads)
    for i in range(num_threads):
      job = mp.Process(target=self.simulate,args=(results, i))
      jobs.append(job)
      job.start()
    [job.join() for job in jobs]
    combined_results = pd.DataFrame(columns=range(self.num_episodes))
    for result in results:
      combined_results = pd.concat((combined_results, result), ignore_index=True)
    return combined_results

  def __copy__(self):
    return Sim(copy(self.world), self.num_episodes, self.num_trials)


if __name__ == "__main__":  
  baseline = {
    "W": RandomModel((0.4, 0.6)),
    "X": ActionModel(("W"), (0, 1)),
    "Z": DiscreteModel(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": DiscreteModel(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  }
  w1 = dict(baseline)
  w1["W"] = RandomModel((0.1, 0.9))
  w9 = dict(baseline)
  w9["W"] = RandomModel((0.9, 0.1))
  z5 = dict(baseline)
  z5["Z"] = DiscreteModel(("X"), {(0,): (0.9, 0.1), (1,): (0.5, 0.5)})
  reversed_z = dict(baseline)
  reversed_z["Z"] = DiscreteModel(("X"), {(0,): (0.25, 0.75), (1,): (0.75, 0.25)})
  reversed_y = dict(baseline)
  reversed_y["Y"] = DiscreteModel(("W", "Z"), {(0, 0): (0, 1), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (1, 0)})

  policies = [Policy.ADJUST]
  # policies = [Policy.NAIVE, Policy.SENSITIVE, Policy.ADJUST]
  pol = Policy.DEAF
  eps = 0.03
  dnc = 0.04
  sn = 10
  start = time.time()
  databank = DataBank(Environment(baseline).domains, Environment(baseline).act_var, Environment(baseline).rew_var)
  
  agents = [
      Agent("0", Environment(baseline), databank, policy, epsilon=eps, div_node_conf=dnc, samps_needed=sn),
      Agent("1", Environment(baseline), databank, policy, eps, dnc, sn),
      Agent("2", Environment(reversed_z), databank, policy, eps, dnc, sn),
      Agent("3", Environment(reversed_z), databank, policy, eps, dnc, sn)
  ]
  for policy in policies:
    databank = DataBank(Environment(baseline).domains, Environment(baseline).act_var, Environment(baseline).rew_var)

    sim = Sim(World(agents), 150, 1)
    sim.multithreaded_sim(ind_var=IV.POL, show=True)
  time = time.time() - start
  mins = time // 60
  sec = time % 60
  print("Time elapsed = {0}:{1}".format(int(mins), sec))
  # plt.savefig("../output/0705-{}agent-{}ep-{}n".format(len(agents), sim.num_episodes, sim.num_trials * mp.cpu_count()))