from query import Product
import util
import gutil
from data import DataBank

class Knowledge():
  def __init__(self, agent, databank):
    self.agent = agent
    self.model = agent.environment.cgm
    self.domains = agent.environment.domains
    self.vars = agent.environment.get_vars()
    self.act_var = agent.environment.get_act_var()
    self.rew_var = agent.environment.get_rew_var()
    self.act_dom = agent.environment.get_act_dom()
    self.rew_dom = agent.environment.get_rew_dom()
    self.databank = databank
    self.databank.add_agent(self.agent)

  def get_recent(self):
    return self.my_data().get_recent()

  def my_data(self):
    return self.databank[self.agent]

  def add_sample(self, sample):
    self.databank[self.agent].append(sample)
  
  def optimal_choice(self, givens={}):
    return self.databank[self.agent].optimal_choice(self.act_dom, self.rew_var, givens)

class KnowledgeNaive(Knowledge): 
  def optimal_choice(self, givens={}):
    return self.databank.all_data().optimal_choice(self.act_dom, self.rew_var, givens)
  
class KnowledgeSensitive(Knowledge):
  def div_nodes(self, agent):
    return self.databank.div_nodes(self.agent, agent)

  def is_divergent_dict(self, agent):
    return self.databank.is_divergent_dict(self.agent, agent)

  def optimal_choice(self, givens={}):
    return self.databank.sensitive_data(self.agent).optimal_choice(self.act_dom, self.rew_var, givens)

class KnowledgeAdjust(KnowledgeSensitive):  
  def get_num_datapoints(self, tf, other):
    return len(gutil.only_dicts_with_givens(self.my_data(), tf[0].get_assignments(tf[0].e())))\
      + len(gutil.only_dicts_with_givens(self.databank[other], tf[1].get_assignments(tf[1].e())))
  
  def transport_formula(self, agent, givens):
    div_nodes = self.div_nodes(agent)
    unformatted_tf = self.model.from_cpts(
        self.model.selection_diagram(
          div_nodes
        ).get_transport_formula(
          self.act_var, self.rew_var, set(givens)
        )
      )
    my_queries = Product()
    other_queries = Product()
    for q in unformatted_tf:
      query_var = q.query_var()
      if query_var in givens or query_var == self.act_var:
        continue
      if query_var in div_nodes:
        my_queries.append(q)
      else:
        other_queries.append(q)
    return Product([my_queries, other_queries]).assign(self.domains).assign(givens)
    
  def solve_transport_formula(self, f, my_data, other_data):
    f = f.deepcopy()
    summation = 0
    for summator in gutil.permutations(f.get_unassigned()):
      f.assign(summator)
      sol1 = f[0].solve(my_data)
      sol2 = f[1].solve(other_data)
      if sol1 is None or sol2 is None:
        return None
      summation += sol1 * sol2
    return summation
    
  def optimal_choice(self, givens={}):
    if len(self.my_data()) < self.agent.samps_needed:
      return Knowledge.optimal_choice(self, givens)
    
    actions = gutil.permutations(self.act_dom)
    rewards = gutil.permutations(self.rew_dom)
    P_data = self.my_data()
    
    action_rewards = {}
    for act in actions:
      act_hash = util.hash_from_dict(act)
      action_rewards[act_hash] = {}
      for rew in rewards:
        action_rewards[act_hash][util.hash_from_dict(rew)] = [[],[]]
    
    for Q_agent, Q_data in self.databank.items():
      transport_formula = self.transport_formula(Q_agent, givens)
      for act in actions:
        transport_formula.assign(act)
        act_hash = util.hash_from_dict(act)
        num_datapoints = self.get_num_datapoints(transport_formula, Q_agent)
        if not num_datapoints:
          continue
        # calculating the transport formula with every action and every agent is slow
        for rew in rewards:
          transport_formula.assign(rew)
          rew_hash = util.hash_from_dict(rew)
          tf_sol = self.solve_transport_formula(transport_formula, P_data, Q_data)
          if tf_sol is None:
            continue
          action_rewards[act_hash][rew_hash][0].append(num_datapoints)
          action_rewards[act_hash][rew_hash][1].append(tf_sol)
    
    weighted_act_rew = gutil.Counter()
    for act in action_rewards:
      for rew in action_rewards[act]:
        reward_prob = 0
        weight_total = sum(action_rewards[act][rew][0])
        if not weight_total: continue
        for i in range(len(action_rewards[act][rew][0])):
          reward_prob += action_rewards[act][rew][1][i] * (action_rewards[act][rew][0][i] / weight_total)
        weighted_act_rew[act] += reward_prob * float(rew.split("=",1)[1])
    return util.dict_from_hash(gutil.max_key(weighted_act_rew))
