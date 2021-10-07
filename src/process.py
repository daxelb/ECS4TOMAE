from agent import SoloAgent, NaiveAgent, SensitiveAgent, AdjustAgent
from world import World
from data import DataBank
from util import printProgressBar
from environment import Environment
import time
from itertools import cycle, combinations_with_replacement
from enums import OTP


class Process:
  def __init__(self, rng, environments, rew_var, is_community, nmc, ind_var, mc_sims, T, ass_perms, num_agents, rand_envs, domains, act_var):
    self.rng = rng
    self.environments = environments
    self.rew_var = rew_var
    self.is_community = is_community
    self.nmc = nmc
    self.ind_var = ind_var
    self.mc_sims = mc_sims
    self.T = T
    self.ass_perms = ass_perms
    self.num_agents = num_agents
    self.rand_envs = rand_envs
    self.domains = domains
    self.act_var = act_var

  def agent_maker(self, name, environment, assignments, agents):
    otp = assignments.pop("otp")
    if otp == OTP.SOLO:
      return SoloAgent(self.rng, name, environment, agents, **assignments)
    elif otp == OTP.NAIVE:
      return NaiveAgent(self.rng, name, environment, agents, **assignments)
    elif otp == OTP.SENSITIVE:
      return SensitiveAgent(self.rng, name, environment, agents, **assignments)
    elif otp == OTP.ADJUST:
      return AdjustAgent(self.rng, name, environment, agents, **assignments)
    else:
      raise ValueError("OTP type %s is not supported." % otp)

  def environment_generator(self):
    nmc = self.nmc if isinstance(
        self.nmc, float) else self.rng.uniform(self.nmc[0], self.nmc[1])
    template = {node: model.randomize(
        self.rng) for node, model in self.environments[0]._assignment.items()}
    base = Environment(template, self.rew_var)
    yield base
    for _ in range(self.num_agents - 1):
      randomized = dict(template)
      for node in base.get_non_act_vars():
        if self.rng.random() < nmc:
          randomized[node] = randomized[node].randomize(self.rng)
      yield Environment(randomized, self.rew_var)

  def world_generator(self):
    # These two lines should NOT be necessary. Need to do testing to make sure.
    ap = list(self.ass_perms)
    self.rng.shuffle(ap)
    # if self.transition_asrs:
    #   assignments = combinations_with_replacement([dict(ass) for ass in ap], self.num_agents)
    # else:
    assignments = [dict(ass) for ass in ap for _ in range(self.num_agents)]
    if not self.is_community:
      self.rng.shuffle(assignments)
    envs = cycle(self.environment_generator()) if self.rand_envs else cycle(self.environments)
    agents = []
    for _ in range(len(self.ass_perms)):
      agents = []
      for i in range(self.num_agents):
        agents.append(self.agent_maker(str(i), next(envs), assignments.pop(), agents))
      yield World(agents, self.T)

  def simulate(self):
    res = [{},{}]
    sim_time = []
    for i in range(self.mc_sims):
      sim_start = time.time()
      for j, world in enumerate(self.world_generator()):
        time_rem = None if not sim_time else \
            (sum(sim_time) / len(sim_time)) * \
            (self.mc_sims - (i + (j / len(self.ass_perms))))
        time_rem_str = '?' if time_rem is None else \
            '%dh %dm   ' % (time_rem // (60 * 60), time_rem // 60 % 60)
        for k in range(self.T):
          world.run_episode(k)
          printProgressBar(
              iteration=i*len(self.ass_perms)+j+(k+1)/self.T,
              total=self.mc_sims * len(self.ass_perms),
              suffix=time_rem_str,
          )
        self.update_process_result(res, world)
      sim_time.append(time.time() - sim_start)
    return res

  def update_process_result(self, res, world):
    raw = (world.cpr, world.poa)
    for i in (0, 1):
      for agent, data in raw[i].items():
        ind_var = agent.get_ind_var_value(self.ind_var)
        if ind_var not in res[i]:
          res[i][ind_var] = [data]
          continue
        res[i][ind_var].append(data)
    return
