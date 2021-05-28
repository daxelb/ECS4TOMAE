from causal_graph import CausalGraph
from knowledge import Knowledge
import util
import random
class Agent:
  def __init__(self, model, domains, action_vars):
    self.knowledge = Knowledge(model, domains)

    self.action_vars = action_vars
    # assumptions: outcome_vars are all leaves in the model,
    # feature_vars are all parents of action_vars
    self.outcome_vars = self.knowledge.get_leaves()
    self.feature_vars = list()
    for node in self.action_vars:
      for parent in self.knowledge.get_parents(node):
        self.feature_vars.append(parent)


  def choose_optimally(self):
    action_domains = {}
    for act_var in self.action_vars:
      action_domains[act_var] = self.knowledge.domains[act_var]
    for choice in random.shuffle(util.combinations(action_domains)):
      if choice not in self.knowledge.obs and choice not in self.knowledge.exp:
        print()


if __name__ == "__main__":
  model = CausalGraph([("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")])
  domains = {"W": (0,1), "X": (0,1), "Y": (0,1), "Z": (0,1)}
  # action_domains = {"X": [0,1]}
  agent0 = Agent(model, domains, "X")
  agent0.knowledge.add_obs([0,1,1,1])
  agent0.knowledge.add_obs([1,1,0,1])
  agent0.knowledge.add_obs([1,0,0,0])
  agent0.knowledge.add_obs([0,0,0,0])
  agent0.knowledge.add_obs([0,1,1,1])
  model.draw_model()
  # print(agent0.knowledge.obs)
  # print(agent0.knowledge.get_conditional_prob({"Y": 1}, {"X": None}, agent0.knowledge.obs))
  # print(agent0.feature_vars)
  # print(agent0.knowledge.domains)
  # print(agent0.get_choice_combinations())

  # print(agent0.knowledge.model.get_distribution())
  # for p in agent0.knowledge.get_model_dist():
  #   print(p)

  sample_query = "P(X)"
  print(sample_query)
  # print()
  for p in util.parse_query(sample_query):
    print(p)
    print(util.prob_with_unassigned(agent0.knowledge.domains, agent0.knowledge.obs, p[0], p[1]))
    print()
  
  parsed_q = util.parse_query(sample_query)[0]
  print(util.kl_divergence(agent0.knowledge.domains, agent0.knowledge.obs, agent0.knowledge.obs, parsed_q[0], parsed_q[1], 2))