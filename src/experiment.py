from data import DataBank
from sim import Sim
from agent import SoloAgent, NaiveAgent, SensitiveAgent, AdjustAgent
from world import World
from copy import copy
from assignment_models import ActionModel, DiscreteModel, RandomModel
from gutil import printProgressBar
from itertools import cycle
from environment import Environment
import plotly.graph_objs as go
import time
from numpy import sqrt, random
import multiprocessing as mp
import pandas as pd

class Experiment:
  def __init__(self, environment_dicts, policy, div_node_conf, asr, epsilon, cooling_rate, num_episodes, num_trials, is_community=False, show=False, save=False, seed=None):
    self.seed = int(random.rand() * 2**32 - 1) if seed is None else seed
    self.rng = random.default_rng(seed)
    self.start_time = time.time()
    self.environments = [Environment(env_dict) for env_dict in environment_dicts]
    self.num_agents = len(self.environments)
    rand_trials = 0
    if policy == "EF":
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
    self.num_cores = mp.cpu_count()
    self.ass_perms = self.get_assignment_permutations()
    self.N = self.num_trials * self.num_agents * len(self.ass_perms) * self.num_cores
    self.is_community = is_community
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
      
  def agent_maker(self, name, environment, databank, assignments):
    policy = assignments.pop("policy")
    if policy == "Solo":
      return SoloAgent(self.rng, name, environment, databank, **assignments)
    elif policy == "Naive":
      return NaiveAgent(self.rng, name, environment, databank, **assignments)
    elif policy == "Sensitive":
      return SensitiveAgent(self.rng, name, environment, databank, **assignments)
    elif policy == "Adjust":
      return AdjustAgent(self.rng, name, environment, databank, **assignments)
    else:
      raise ValueError("Policy type " + policy + " is not supported.")
      
  def world_generator(self):
    agent_assignments = [dict(ass) for ass in self.ass_perms * int(self.N / len(self.ass_perms))]
    if not self.is_community:
      self.rng.shuffle(agent_assignments)
    env_cycle = cycle(self.environments)
    worlds = []
    for thread_num in range(self.num_cores):
      worlds.append([])
      for _ in range(len(self.ass_perms) * self.num_trials):
        databank = self.environments[0].create_empty_databank()
        agents = [self.agent_maker(str(i), next(env_cycle), databank, agent_assignments.pop()) for i in range(self.num_agents)]
        worlds[thread_num].append(World(agents, self.is_community))
    return worlds
  
  def multithreaded_sim(self, worlds):
    jobs = []
    results = mp.Manager().list([None] * self.num_threads)
    for i, thread_worlds in enumerate(worlds):
      job = mp.Process(target=self.simulate, args=(thread_worlds, results, i))
      jobs.append(job)
      job.start()
    [job.join() for job in jobs]
    return results
  
  def simulate(self, worlds, results, index):
    thread_result = {}
    for i, world in enumerate(worlds):
      for j in self.num_episodes:
        world.run_once()
        printProgressBar(i+((j+1)/self.num_episodes), self.num_trials)
      for ind_var, data in world.pseudo_cum_regret:
        if ind_var not in thread_result:
          thread_result[ind_var] = pd.DataFrame(columns=range(self.num_episodes))
        pd.concat((thread_result[ind_var], data), ignore_index=True)
    results[index] = thread_result
  
  def combine_results(self, results):
    combined_results = {}
    for result in results:
      for ind_var in result:
        if ind_var not in combined_results:
          combined_results[ind_var] = pd.DataFrame(columns=range(self.num_episodes))
        for trial_data in result[ind_var]:
          combined_results[ind_var] = pd.concat((combined_results[ind_var], trial_data), ignore_index=True)
    return combined_results
  
  def get_plot(self, results):
    figure = []
    x = list(range(self.num_episodes))
    for i, ind_var in enumerate(results):
      line_hue = str(int(360 * (i / len(results))))
      df = results[ind_var]
      y = df.mean(axis=0)
      sqrt_variance = sqrt(df.var(axis=0))
      y_upper = y + sqrt_variance
      y_lower = y - sqrt_variance
      line_color = "hsla(" + line_hue + ",100%,50%,1)"
      error_band_color = "hsla(" + line_hue + ",100%,50%,0.125)"
      figure.extend([
      go.Scatter(
        name=ind_var,
        x=x,
        y=y,
        line=dict(color=line_color),
        mode='lines',
      ),
      go.Scatter(
        name=ind_var+"-upper",
          x=x,
          y=y_upper,
          mode='lines',
          marker=dict(color=error_band_color),
          line=dict(width=0),
          # showlegend=False,
      ),
      go.Scatter(
          name=ind_var+"-lower",
          x=x,
          y=y_lower,
          marker=dict(color=error_band_color),
          line=dict(width=0),
          mode='lines',
          fillcolor=error_band_color,
          fill='tonexty',
          # showlegend=False,
      )
    ])
    plotly_fig = go.Figure(figure)
    plotly_fig.update_layout(
      yaxis_title="Pseudo Cumulative Regret",
      xaxis_title="Episodes",
      title="graph pog",
    )
    return plotly_fig
    
  def display_and_save(self, plot):
    elapsed_time = time.time() - self.start_time
    print("\n\nTime elapsed: {:02d}:{:02d}:{:05.2f}".format(
      int(elapsed_time // (60 * 60)),
      int((elapsed_time // 60)),
      elapsed_time % 60
    ))
    print("Seed:", self.seed)
    if self.show:
      plot.show()
    if self.save:
      date = time.strftime("%m%d", time.gmtime())
      file_name = "../output/{}-{}agent-{}ep-{}n".format(date, len(self.environments), self.num_episodes, self.num_trials * mp.cpu_count())
      plot.write_html(file_name + ".html")
      self.saved_data.to_csv(file_name + "-last_episode_data.csv")
      
  def run(self):
    results = self.combine_results(
      self.multithreaded_sim(
        self.world_generator()
      )
    )
    self.display_and_save(self.get_plot(results))
    
    
    

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
    policy=("Solo", "Naive"), #"Sensitive", "Adjust"
    asr="TS",
    epsilon=0.075,
    cooling_rate=0.05,
    div_node_conf=0.04, 
    num_episodes=100,
    num_trials=15,
    is_community=True,
    show=True,
    save=False,
    seed=None
  )
  print(len(experiment.world_generator()[0]))
  # experiment.run()
