from enum import Enum
from assignment_models import ActionModel, DiscreteModel, RandomModel
from environment import Environment
import plotly.graph_objs as go
import time
from numpy import random
import multiprocessing as mp
from pandas import DataFrame, ExcelWriter
from os import mkdir
from json import dump
from enums import OTP, ASR
from process import Process

class Sim:
  def __init__(self, environment_dicts, otp, tau, asr, T, mc_sims, EG_epsilon=0, EF_rand_trials=0, ED_cooling_rate=0, is_community=False, rand_envs=False, node_mutation_chance=0, show=True, save=False, seed=None):
    self.seed = int(random.rand() * 2**32 - 1) if seed is None else seed
    self.rng = random.default_rng(self.seed)
    self.start_time = time.time()
    self.rand_envs = rand_envs
    self.nmc = node_mutation_chance
    self.environments = [Environment(env_dict) for env_dict in environment_dicts]
    self.num_agents = len(self.environments)
    self.assignments = {
      "otp": otp,
      "tau": tau,
      "asr": asr,
      "epsilon": EG_epsilon,
      "rand_trials": EF_rand_trials,
      "cooling_rate": ED_cooling_rate,
    }
    if isinstance(asr, str):
      if asr != ASR.EG:
        del self.assignments["epsilon"]
      if asr != ASR.EF:
        del self.assignments["rand_trials"]
      if asr != ASR.ED:
        del self.assignments["cooling_rate"]
    elif isinstance(asr, (tuple, list, set)):
      if ASR.EG not in asr:
        del self.assignments["epsilon"]
      if ASR.EF not in asr:
        del self.assignments["rand_trials"]
      if ASR.ED not in asr:
        del self.assignments["cooling_rate"]
    self.ind_var = self.get_ind_var()
    self.T = T
    self.mc_sims = mc_sims
    self.num_threads = mp.cpu_count()
    self.ass_perms = self.get_assignment_permutations()
    self.is_community = is_community
    self.show = show
    self.save = save
    self.data_cpr = {}
    self.data_poa = {}
    self.last_episode_cpr = DataFrame()
    self.last_episode_poa = DataFrame()
    self.values = self.get_values(locals())
    self.domains = self.environments[0].get_domains()
    self.act_var = self.environments[0].get_act_var()
    self.rew_var = self.environments[0].get_rew_var()
  
  def multithreaded_sim(self):
    jobs = []
    results = mp.Manager().list([None] * self.num_threads)
    for i in range(self.num_threads):
      job = mp.Process(target=self.sim_process, args=(results, i))
      jobs.append(job)
      job.start()
    [job.join() for job in jobs]
    return results

  def process_args(self, index):
    return {
      'rng': random.default_rng(abs(self.seed - index)),
      'environments': self.environments,
      'rew_var': self.rew_var,
      'is_community': self.is_community,
      'nmc': self.nmc,
      'ind_var': self.ind_var,
      'mc_sims': self.mc_sims,
      'T': self.T,
      'ass_perms': self.ass_perms,
      'num_agents': self.num_agents,
      'rand_envs': self.rand_envs,
      'domains': self.domains,
      'act_var': self.act_var
    }

  def sim_process(self, results, index):
    proc = Process(**self.process_args(index))
    results[index] = proc.simulate()
    return

  def combine_results(self, process_results):
    results = [{},{}]
    for pr in process_results:
      for i in range(len(results)):
        for ind_var, res in pr[i].items():
          if ind_var not in results[i]:
            results[i][ind_var] = res
            continue
          results[i][ind_var].extend(res)
    return results

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

  def get_plot(self, results, plot_title, yaxis_title):
    figure = []
    x = list(range(self.T))
    for i, ind_var in enumerate(sorted(results)):
      line_name = str(ind_var)
      line_hue = str(int(360 * (i / len(results))))
      df = DataFrame(results[ind_var])
      if yaxis_title == "Cumulative Pseudo Regret":
        self.last_episode_cpr.insert(0, line_name, df.iloc[:, -1])
        self.data_cpr[ind_var] = df
      else:
        self.last_episode_poa.insert(0, line_name, df.iloc[:, -1])
        self.data_poa[ind_var] = df
      y = df.mean(axis=0, numeric_only=True)
      sem = df.sem(axis=0, numeric_only=True)
      y_upper = y + sem
      y_lower = y - sem
      line_color = "hsla(" + line_hue + ",100%,40%,1)"
      error_band_color = "hsla(" + line_hue + ",100%,40%,0.125)"
      figure.extend([
          go.Scatter(
              name=line_name,
              x=x,
              y=y,
              line=dict(color=line_color, width=3),
              mode='lines',
          ),
          go.Scatter(
              name=line_name+"-upper",
              x=x,
              y=y_upper,
              mode='lines',
              marker=dict(color=error_band_color),
              line=dict(width=0),
              showlegend=False,
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
              showlegend=False,
          )
      ])
    plotly_fig = go.Figure(figure)
    plotly_fig.update_layout(
        font=dict(size=18),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        yaxis_title=yaxis_title,
        xaxis_title="Trial",
        # title=plot_title,
    )
    return plotly_fig

  
  def get_cpr_plot(self, results, desc):
    plot_title = "Community CPR of " + desc if self.is_community else "Mean Agent CPR of " + desc
    return self.get_plot(results, plot_title, "Cumulative Pseudo Regret")
  
  def get_poa_plot(self, results, desc):
    plot_title = "POA of " + desc
    return self.get_plot(results, plot_title, "Probability of Optimal Action")

  def display_and_save(self, results, desc):
    cpr_plot = self.get_cpr_plot(results[0], desc)
    poa_plot = self.get_poa_plot(results[1], desc)
    elapsed_time = time.time() - self.start_time
    print_info = "Time Elapsed: {}h {}m {}s".format(\
      int(elapsed_time // (60 * 60)),\
      int(elapsed_time // 60 % 60),\
      int(elapsed_time % 60)
    )
    print(f'{print_info}{" " * (70 - len(print_info))}')

    if self.show:
      cpr_plot.show()
      poa_plot.show()
    if self.save:
      file_name = "{}_N{}".format(desc, self.get_N())
      dir_path = "../output/%s" % file_name
      mkdir(dir_path)
      cpr_plot.write_html(dir_path + "/cpr.html")
      poa_plot.write_html(dir_path + "/poa.html")
      self.last_episode_cpr.to_csv(dir_path + "/last_episode_cpr.csv")
      with ExcelWriter(dir_path + '/cpr.xlsx') as writer:  # doctest: +SKIP
        for ind_var, df in self.data_cpr.items():
          sheet_name = str(ind_var) if ind_var else 'Sheet1'
          df.to_excel(writer, sheet_name=sheet_name)
      with ExcelWriter(dir_path + '/poa.xlsx') as writer:  # doctest: +SKIP
        for ind_var, df in self.data_poa.items():
          sheet_name = str(ind_var) if ind_var else 'Sheet1'
          df.to_excel(writer, sheet_name=sheet_name)
      with open(dir_path + '/values.json', 'w') as outfile:
        dump(self.values, outfile)
      
  def run(self, desc=None):
    if desc:
      print(desc)
    print("seed=%d | N=%d" % (self.seed, self.get_N()))
    results = self.combine_results(self.multithreaded_sim())
    self.display_and_save(results, desc)
    
  def get_N(self):
    return self.num_threads * self.mc_sims * self.num_agents
  
  def get_values(self, locals):
    values = {key: val for key, val in locals.items() if key != 'self'}
    values["otp"] = values["otp"].value if isinstance(values["otp"], Enum) else [e.value for e in values["otp"]]
    values["asr"]    = values["asr"].value    if isinstance(values["asr"], Enum)    else [e.value for e in values["asr"]]
    parsed_env_dicts = []
    for env in values["environment_dicts"]:
      parsed_env = {}
      for node, model in env.items():
        parsed_env[node] = str(model)
      parsed_env_dicts.append(parsed_env)
    values["environment_dicts"] = tuple(parsed_env_dicts)
    values["seed"] = self.seed
    return values

if __name__ == "__main__":
  baseline = {
    "Z": RandomModel((0.5, 0.5)),
    "X": ActionModel(("Z"), (0, 1)),
    "W": DiscreteModel(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": DiscreteModel(("Z", "W"), {(0, 0): (0.8, 0.2), (0, 1): (0.5, 0.5), (1, 0): (0.5, 0.5), (1, 1): (0.2, 0.8)})
  }
  reversed_w = dict(baseline)
  reversed_w["W"] = DiscreteModel(("X"), {(0,): (0.25, 0.75), (1,): (0.75, 0.25)})

  experiment = Sim(
    environment_dicts=(baseline, reversed_w, baseline, reversed_w),
    otp=OTP.ADJUST,
    asr=ASR.TS,
    T=3000,
    mc_sims=30,
    tau=0.05,
    EG_epsilon=0.02,
    EF_rand_trials=25,
    ED_cooling_rate=0.96,
    is_community=False,
    rand_envs=True,
    node_mutation_chance=(0.2,0.8),
    show=True,
    save=True,
    seed=None
  )
  experiment.run(desc="skip agents with S-node on W and Y")
