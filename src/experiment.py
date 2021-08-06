from data import DataBank
from sim import Sim
from agent import SoloAgent, NaiveAgent, SensitiveAgent, AdjustAgent
from world import World
from copy import copy
from assignment_models import ActionModel, DiscreteModel, RandomModel
from environment import Environment
import plotly.graph_objs as go
import time
from numpy import sqrt
from enums import Policy, ASR
import multiprocessing as mp
import pandas as pd

class Experiment:
  def __init__(self, environment_dicts, policy, div_node_conf, asr, epsilon, cooling_rate, num_episodes, num_trials, show=False, save=False):
    self.environments = [Environment(env_dict) for env_dict in environment_dicts]
    rand_trials = 0
    if policy == ASR.EPSILON_FIRST:
      if isinstance(epsilon, (tuple, list, set)):
        rand_trials = [int(num_episodes * eps) for eps in epsilon]
        epsilon = 0
      else:
        rand_trials = int(num_episodes * epsilon)
    self.assignments = {
      "policy": policy,
      "div_node_conf": div_node_conf,
      "asr": asr,
      "epsilon": epsilon,
      "rand_trials": rand_trials,
      "cooling_rate": cooling_rate,
    }
    self.ind_var = self.get_ind_var()
    self.num_episodes = num_episodes
    self.num_trials = num_trials
    self.show = show
    self.save = save
    self.saved_data = pd.DataFrame(index=range(self.num_trials * self.num_episodes))
    
    
  def get_ind_var(self):
    ind_var = None
    for var, assignment in self.assignments.items():
      if isinstance(assignment, (list, tuple, set)):
        assert ind_var is None
        ind_var = var
    return ind_var
  
  def get_assignment_permutations(self):
    if self.ind_var is None:
      return [self.assignments]
    permutations = []
    for ind_var_assignment in self.assignments[self.ind_var]:
      permutation = dict(self.assignments)
      permutation[self.ind_var] = ind_var_assignment
      permutations.append(permutation)
    return permutations
  
  def sim(self, line_name, line_hue, assignment_permutation):
    db = DataBank(self.environments[0].get_domains(), self.environments[0].get_act_var(), self.environments[0].get_rew_var())
    agents = []
    policy = assignment_permutation.pop("policy")
    for i, environment in enumerate(self.environments):
      if policy == Policy.SOLO:
        agents.append(SoloAgent(str(i), environment, db, **assignment_permutation))
      elif policy == Policy.NAIVE:
        agents.append(NaiveAgent(str(i), environment, db, **assignment_permutation))
      elif policy == Policy.SENSITIVE:
        agents.append(SensitiveAgent(str(i), environment, db, **assignment_permutation))
      elif policy == Policy.ADJUST:
        agents.append(AdjustAgent(str(i), environment, db, **assignment_permutation))
    sim = Sim(World(agents), self.num_episodes, self.num_trials)
    result = sim.multithreaded_sim()
    self.saved_data[line_name] = result.iloc[:,-1:]
    x = list(range(self.num_episodes))
    y = result.mean(axis=0)
    variance = result.var(axis=0)
    y_upper = y + sqrt(variance)
    y_lower = y - sqrt(variance)
    line_color = "hsla(" + line_hue + ",100%,50%,1)"
    error_band_color = "hsla(" + line_hue + ",100%,50%,0.125)"
    return [
      go.Scatter(
        name=line_name,
        x=x,
        y=y,
        line=dict(color=line_color),
        mode='lines',
      ),
      go.Scatter(
        name=line_name+"-upper",
          x=x,
          y=y_upper,
          mode='lines',
          marker=dict(color=error_band_color),
          line=dict(width=0),
          # showlegend=False,
      ),
      go.Scatter(
          name=line_name+"-lower",
          x=x,
          y=y_lower,
          marker=dict(color=error_band_color),
          line=dict(width=0),
          mode='lines',
          fillcolor=error_band_color,
          fill='tonexty',
          # showlegend=False,
      )
    ]
      
  def run(self):
    start_time = time.time()
    figure = []
    permutations = self.get_assignment_permutations()
    for i, permutation in enumerate(permutations):
      line_name = ""
      if self.ind_var in ("policy", "asr"):
        line_name = permutation[self.ind_var].value
      elif self.ind_var is not None:
        line_name = str(permutation[self.ind_var])
      line_hue = str(int(360 * (i / len(permutations))))
      if i:
        print()
      print(line_name)
      figure.extend(self.sim(line_name, line_hue, permutation))
    plotly_fig = go.Figure(figure)
    plotly_fig.update_layout(
      yaxis_title="Pseudo Cumulative Regret",
      xaxis_title="Episodes",
      title="graph pog",
    )
    elapsed_time = time.time() - start_time
    hrs = elapsed_time // (60 * 60)
    mins = (elapsed_time // 60) 
    sec = elapsed_time % 60
    print("\nTime elapsed = {:02d}:{:02d}:{:.2f}".format(int(hrs), int(mins), sec))
    if self.show:
      plotly_fig.show()
    if self.save:
      date = time.strftime("%m%d", time.gmtime())
      file_name = "../output/{}-{}agent-{}ep-{}n".format(date, len(self.environments), self.num_episodes, self.num_trials * mp.cpu_count())
      plotly_fig.write_html(file_name + ".html")
      self.saved_data.to_csv(file_name + "-last_episode_data.csv")

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

  experiment = Experiment(
    environment_dicts=(baseline, baseline, reversed_z, reversed_z),
    policy=(Policy.SOLO, Policy.NAIVE, Policy.SENSITIVE, Policy.ADJUST),
    asr=ASR.THOMPSON_SAMPLING,
    epsilon=0.075,
    cooling_rate=0.05,
    div_node_conf=0.04, 
    num_episodes=275,
    num_trials=15,
    show=True,
    save=True
  )
  experiment.run()
