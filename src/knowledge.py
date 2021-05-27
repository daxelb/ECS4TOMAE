class Knowledge():
  def __init__(self, model, domains):
    self.model = model
    self.domains = domains
    self.most_recent = None
    self.observations = self.experiments = {}
    for var_name in self.get_observable():
      self.observations[var_name] = []
      self.experiments[var_name] = []

  def get_observable(self):
    return self.model.get_observable()

  def get_parents(self, node):
    return self.model.get_parents(node)

  def get_children(self, node):
    return self.model.get_children(node)

  def get_exogenous(self):
    return self.model.get_exogenous()

  def get_endogenous(self):
    return self.model.get_endogenous()
  
  def get_leaves(self):
    return self.model.get_leaves()

  def draw_model(self, v=False):
    self.model.draw().render('output/causal-model.gv', view=v)
  
  def add_obs(self, sample):
    observable_vars = self.get_observable()
    if len(observable_vars) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(observable_vars, sample))
    for var_name in observable_vars:
      self.observations[var_name].append(sample[var_name])
    self.most_recent = sample
  
  def add_exp(self, sample):
    observable_vars = self.get_observable()
    if len(observable_vars) != len(sample):
      print("Error adding sample to agent's dataset.")
    # if sample is a list, format correctly as dict
    if type(sample) is list:
      sample = dict(zip(observable_vars, sample))
    for var_name in observable_vars:
      self.experiments[var_name].append(sample[var_name])
    self.most_recent = sample

  def get_model_dist(self):
    return util.parse_query(self.model.get_distribution())