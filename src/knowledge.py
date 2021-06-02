import util
class Knowledge():
  def __init__(self, model, domains, action_vars):
    self.model = model
    self.domains = domains
    self.action_vars = action_vars
    self.most_recent = None
    self.obs = list()
    self.exp = list()
  
  def add_obs(self, sample):
    if len(self.model.observed_variables) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(self.model.observed_variables, sample))
    # for var_name in observable_vars:
    self.obs.append(sample)
      # self.obs[var_name].append(sample[var_name])
    self.most_recent = sample
  
  def add_exp(self, sample):
    if len(self.model.observed_variables) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(self.model.observed_variables, sample))
    self.exp.append(sample)
    # for var_name in observable_vars:
      # self.exp[var_name].append(sample[var_name])
    self.most_recent = sample

  def get_useful_data(self):
    for var in self.action_vars:
      if self.model.do(var).get_parents(var) != self.model.get_parents(var):
        return self.exp
    return self.obs + self.exp

  def get_model_dist(self):
    return util.parse_query(self.model.get_distribution())
