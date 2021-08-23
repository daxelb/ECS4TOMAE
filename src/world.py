from agent import SensitiveAgent, AdjustAgent
from util import only_given_keys

class World:
  def __init__(self, agents, T,is_community=False):
    self.agents = agents
    self.is_community = is_community
    self.pseudo_cum_regret = {a: [0] * T for a in self.agents}
    self.optimal_action = {a: [0] * T for a in self.agents}
    self.has_sensitive = any([isinstance(a, (SensitiveAgent, AdjustAgent)) for a in self.agents])
    self.databank = agents[0].databank

  def run_episode(self, ep):
    [a.act() for a in self.agents]
    if self.has_sensitive:
      self.databank.update_divergence()
    self.update_pseudo_regret(ep)
    return
  
  def update_pseudo_regret(self, ep):
    for a in self.agents:
      recent = a.get_recent()
      rew_received = recent[a.reward_var]
      feature_assignments = only_given_keys(recent, a.environment.feat_vars)
      rew_optimal = a.environment.get_optimal_reward(feature_assignments)
      curr_regret = self.pseudo_cum_regret[a][ep-1]
      new_regret = curr_regret + (rew_optimal - rew_received)
      self.pseudo_cum_regret[a][ep] = new_regret
      optimal_actions = a.environment.get_optimal_actions(feature_assignments)
      self.optimal_action[a][ep] = 1.0 if recent[a.action_var] in optimal_actions else 0.0
    return
  
  def __reduce__(self):
    return (self.__class__, (self.agents, self.T, self.is_community))
