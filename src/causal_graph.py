from causalgraphicalmodels import CausalGraphicalModel
class CausalGraph(CausalGraphicalModel):
  def __init__(self, edges, latent_edges=None):
    nodes = self.get_nodes_from_edges(edges, latent_edges)
    super(CausalGraph, self).__init__(nodes, edges, latent_edges)

  def get_nodes_from_edges(self, obs_edges, latent_edges):
    edges = obs_edges
    if latent_edges:
      edges.extend(e for e in latent_edges if e not in edges)
    nodes = list()
    for tup in edges:
      for i in range(len(tup)):
        if tup[i] not in nodes:
          nodes.append(tup[i])
    return nodes

  def get_observable(self):
    return sorted(list(self.dag.nodes))

  def get_parents(self, node):
    return sorted(list(self.dag.predecessors(node)))

  def get_children(self, node):
    return sorted(list(self.dag.successors(node)))

  def get_exogenous(self):
    return sorted([n for n in self.dag.nodes if not self.get_parents(n)])

  def get_endogenous(self):
    return sorted([n for n in self.dag.nodes if self.get_parents(n)])

  def get_leaves(self):
    leaves = list()
    for i in self.dag.nodes:
      if not len(list(self.dag.successors(i))):
        leaves.append(i)
    return leaves

  def draw_model(self, v=False):
    self.draw().render('output/causal-model.gv', view=v)