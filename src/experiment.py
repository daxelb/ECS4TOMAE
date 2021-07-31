from data import DataBank
from sim import Sim
from agent import Agent
from world import World
from copy import copy
from assignment_models import ActionModel, DiscreteModel, RandomModel
from environment import Environment
import plotly.graph_objs as go
import time
from enums import Policy


class Experiment:
  def __init__(self, environments, pol, eps, dnc, sn, num_episodes, num_trials, show=False, save=False):
    self.environments = environments
    self.pol = pol
    self.eps = eps
    self.dnc = dnc
    self.sn = sn
    self.assignments = (pol, eps, dnc, sn)
    self.ind_var_i = self.get_ind_var_index()
    self.num_episodes = num_episodes
    self.num_trials = num_trials
    self.show = show
    self.save = save
    
    
  def get_ind_var_index(self):
    ind_var_index = -99
    for i, a in enumerate(self.assignments):
      if isinstance(a, (list, tuple, set)):
        assert ind_var_index == -99
        ind_var_index = i
    return ind_var_index
  
  def get_assignment_permutations(self):
    if self.ind_var_i == -99:
      return [self.assignments]
    permutations = []
    for ind_var_assignment in self.assignments[self.ind_var_i]:
      permutation = list(self.assignments)
      permutation[self.ind_var_i] = ind_var_assignment
      permutations.append(permutation)
    return permutations
  
  def sim(self, line_name, ass_perm):
    db = DataBank(self.environments[0].get_domains(), self.environments[0].get_act_var(), self.environments[0].get_rew_var())
    pol = ass_perm[0]
    eps = ass_perm[1]
    dnc = ass_perm[2]
    sn = ass_perm[3]
    agents = [Agent(str(i), env, db, pol, eps, dnc, sn) for i, env in enumerate(self.environments)]
    sim = Sim(World(agents), self.num_episodes, self.num_trials)
    result = sim.multithreaded_sim()
    x = list(range(self.num_episodes))
    y = result.mean(axis=0)
    y_upper = result.max(axis=0)
    y_lower = result.min(axis=0)
    return [
      go.Scatter(
        name=line_name,
        x=x,
        y=y,
        mode='lines'
      ),
      go.Scatter(
        name=line_name+"-upper",
        x=x,
        y=y_upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False
      ),
      go.Scatter(
        name=line_name+"-lower",
        x=x,
        y=y_lower,
        mode='lines',
        line=dict(width=0),
        showlegend=False
      )
    ]
      
  def run(self):
    start_time = time.time()
    figure = []
    for permutation in self.get_assignment_permutations():
      line_name = str(permutation[self.ind_var_i]) if self.ind_var_i else permutation[self.ind_var_i].value
      figure.extend(self.sim(line_name, permutation))
    plotly_fig = go.Figure(figure)
    plotly_fig.update_layout(
      yaxis_title="Pseudo Cumulative Regret",
      title="BlahBlahBlah",
      hovermode="x"
    )
    elapsed_time = time.time() - start_time
    hrs = elapsed_time // (60 * 60)
    mins = elapsed_time // 60
    sec = elapsed_time % 60
    print("Time elapsed = {0}:{1}:{2}".format(int(hrs), int(mins), sec))
    if self.show:
      plotly_fig.show()

if __name__ == "__main__":  
  baseline = Environment({
    "W": RandomModel((0.4, 0.6)),
    "X": ActionModel(("W"), (0, 1)),
    "Z": DiscreteModel(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
    "Y": DiscreteModel(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
  })
  w1 = copy(baseline)
  w1["W"] = RandomModel((0.1, 0.9))
  w9 = copy(baseline)
  w9["W"] = RandomModel((0.9, 0.1))
  z5 = copy(baseline)
  z5["Z"] = DiscreteModel(("X"), {(0,): (0.9, 0.1), (1,): (0.5, 0.5)})
  reversed_z = copy(baseline)
  reversed_z["Z"] = DiscreteModel(("X"), {(0,): (0.25, 0.75), (1,): (0.75, 0.25)})
  reversed_y = copy(baseline)
  reversed_y["Y"] = DiscreteModel(("W", "Z"), {(0, 0): (0, 1), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (1, 0)})

  environments = [baseline, baseline, reversed_z, reversed_z]
  pol = [Policy.DEAF, Policy.NAIVE, Policy.SENSITIVE, Policy.ADJUST]
  eps = 0.03
  dnc = 0.04
  sn = 10
  num_episodes = 250
  num_trials = 1
  experiment = Experiment(environments, pol, eps, dnc, sn, num_episodes, num_trials, show=True, save=False)
  experiment.run()
  # plt.savefig("../output/0705-{}agent-{}ep-{}n".format(len(agents), sim.num_episodes, sim.num_trials * mp.cpu_count()))