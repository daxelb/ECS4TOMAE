# Besides some added methods, most of this class was written by Iain Barr (ijmbarr on GitHub)
# from his public repository, causalgraphicalmodels
# The code has been imported and modified into this project for ease/consistency

import networkx as nx
import graphviz
from itertools import combinations, chain
from collections.abc import Iterable
import gutil

class CausalGraph:
  """
  Causal Graphical Models
  """
  def __init__(self, nodes, edges, latent_edges=None, set_nodes=None, s_node_children=set()):
    """
    Create CausalGraph

    Arguments
    ---------
    nodes: list[node:str]

    edges: list[tuple[node:str, node:str]]

    latent_edges: list[tuple[node:str, node:str]] or None

    set_nodes: list[node:str] or None
    """
    self.set_nodes = frozenset() if set_nodes is None else frozenset(sorted(set_nodes))
    self.latent_edges = frozenset() if latent_edges is None else frozenset(sorted(latent_edges))

    self.dag = nx.DiGraph()
    self.dag.add_nodes_from(nodes)
    self.dag.add_edges_from(edges)

    s_nodes = []
    for n in s_node_children:
      new_node = "Snode_{}".format(n)
      self.dag.add_node(new_node)
      self.dag.add_edge(new_node, n)
      s_nodes.append(new_node)
    self.s_nodes = sorted(s_nodes)

    # Add latent connections to the graph
    self.observed_vars = sorted(nodes + s_nodes)
    self.unobserved_var_edges = dict()
    unobserved_vars = []
    unobserved_var_counter = 0
    for n1, n2 in self.latent_edges:
      new_node = "UC_{}".format(unobserved_var_counter)
      unobserved_var_counter += 1
      self.dag.add_node(new_node)
      self.dag.add_edge(new_node, n1)
      self.dag.add_edge(new_node, n2)
      unobserved_vars.append(new_node)
      self.unobserved_var_edges[new_node] = (n1, n2)
    self.unobserved = frozenset(sorted(unobserved_vars))

    assert nx.is_directed_acyclic_graph(self.dag)

    self.graph = self.dag.to_undirected()

  def __repr__(self):
    variables = ", ".join(map(str, sorted(self.observed_vars)))
    return ("{classname}({vars})"
        .format(classname=self.__class__.__name__,
            vars=variables))

  # def has_latent_parents(self, node):
  #   return len(self.do(node).get_parents(node)) == len(self.get_parents(node))
  
  def pa(self, node):
    return sorted([n for n in self.get_parents(node) + [node] if n in self.observed_vars])
  
  def an(self, node):
    return sorted([n for n in self.get_ancestors(node) + [node] if n in self.observed_vars])
  
  def de(self, node):
    return sorted([n for n in self.get_descendants(node) + [node] if n in self.observed_vars])

  def get_ancestors(self, node):
    return sorted(self.get_ancestors_helper(self.get_parents(node)))

  def get_ancestors_helper(self, frontier=[], visited=[]):
    for node in frontier:
      if node not in visited:
        visited.append(node)
        self.get_ancestors_helper(self.get_parents(node), visited)
    return visited
  
  def get_descendants(self, node):
    return sorted(self.get_descendants_helper(self.get_children(node)))
  
  def get_descendants_helper(self, frontier=[], visited=[]):
    for node in frontier:
      if node not in visited:
        visited.append(node)
        self.get_descendants_helper(self.get_children(node), visited)
    return visited

  def get_observable(self):
    return sorted(list(self.observed_vars))

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

  def draw(self):
    """
    dot file representation of the CGM.
    """
    dot = graphviz.Digraph()

    for node in self.observed_vars:
      if node in self.set_nodes:
        dot.node(node, node, {"shape": "ellipse", "peripheries": "2"})
      else:
        dot.node(node, node, {"shape": "ellipse"})

    for a, b in self.dag.edges():
      if a in self.observed_vars and b in self.observed_vars:
        dot.edge(a, b)

    for n, (a, b) in self.unobserved_var_edges.items():
      dot.node(n, _attributes={"shape": "point"})
      dot.edge(n, a, _attributes={"style": "dashed"})
      dot.edge(n, b, _attributes={"style": "dashed"})

    return dot

  def get_distribution(self):
    """
    Returns a string representing the factorized distribution implied by
    the CGM.
    """
    products = []
    for node in nx.topological_sort(self.dag):
      if node in self.set_nodes:
        continue

      parents = list(self.dag.predecessors(node))
      if not parents:
        p = "P({})".format(node)
      else:
        formatted_parents = []
        [formatted_parents.append("do({})".format(parent)) for parent in sorted(
          parents) if parent in self.set_nodes]
        [formatted_parents.append(str(parent)) for parent in sorted(parents) if parent not in self.set_nodes]
        p = "P({}|{})".format(node, ",".join(formatted_parents))
      products.append(p)
    return "".join(products)

  def get_node_distributions(self):
    product = []
    for node in nx.topological_sort(self.dag):
      parents = self.get_parents(node)
      query = {node: None}
      givens = {}
      for parent in parents:
        givens[parent] = None
      product.append((query, givens))
    return product
  
  def get_non_s_nodes(self):
    return [n for n in self.observed_vars if n not in self.s_nodes]

  def get_edges(self):
    return [
      (a, b)
      for a, b in self.dag.edges()
      if a in self.observed_vars
      and b in self.observed_vars]
    
  def get_latent_edges(self):
    return [
      (a, b)
      for a, b in self.latent_edges
      if a not in self.set_nodes
      and b not in self.set_nodes
    ]
    
  def get_s_node_children(self):
    children = []
    [children.extend(self.get_children(n)) for n in self.s_nodes]
    gutil.remove_dupes(children)
    return sorted(children)
    
  def do(self, node):
    """
    Apply intervention on node to CGM
    """
    assert node in self.observed_vars
    set_nodes = self.set_nodes | frozenset([node])
    edges = [e for e in self.get_edges() if e[1] != node]
    latent_edges = [e for e in self.get_latent_edges() if e[0] != node and e[1] != node]
    return CausalGraph(
      nodes=self.get_non_s_nodes(), edges=edges,
      latent_edges=latent_edges, set_nodes=set_nodes, s_node_children=self.get_s_node_children())

  def selection_diagram(self, s_node_children):
    return CausalGraph(
      nodes=self.get_non_s_nodes(), edges=self.get_edges(),
      latent_edges=self.get_latent_edges(), set_nodes=self.set_nodes,
      s_node_children=s_node_children)

  def _check_d_separation(self, path, zs=None):
    """
    Check if a path is d-separated by set of variables zs.
    """
    zs = _variable_or_iterable_to_set(zs)

    if len(path) < 3:
      return False

    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
      structure = self._classify_three_structure(a, b, c)

      if structure in ("chain", "fork") and b in zs:
        return True

      if structure == "collider":
        descendants = (nx.descendants(self.dag, b) | {b})
        if not descendants & set(zs):
          return True

    return False

  def _classify_three_structure(self, a, b, c):
    """
    Classify three structure as a chain, fork or collider.
    """
    if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
      return "chain"

    if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
      return "chain"

    if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
      return "collider"

    if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
      return "fork"

    raise ValueError("Unsure how to classify ({},{},{})".format(a, b, c))

  def is_d_separated(self, x, y, zs=None):
    """
    Is x d-separated from y, conditioned on zs?
    """
    zs = _variable_or_iterable_to_set(zs)
    assert x in self.observed_vars
    assert y in self.observed_vars
    assert all([z in self.observed_vars for z in zs])

    paths = nx.all_simple_paths(self.graph, x, y)
    return all(self._check_d_separation(path, zs) for path in paths)

  def get_all_independence_relationships(self):
    """
    Returns a list of all pairwise conditional independence relationships
    implied by the graph structure.
    """
    conditional_independences = []
    for x, y in combinations(self.observed_vars, 2):
      remaining_variables = set(self.observed_vars) - {x, y}
      for cardinality in range(len(remaining_variables) + 1):
        for z in combinations(remaining_variables, cardinality):
          if self.is_d_separated(x, y, frozenset(z)):
            conditional_independences.append((x, y, set(z)))
    return conditional_independences

  def get_all_backdoor_paths(self, x, y):
    """
    Get all backdoor paths between x and y
    """
    return [
      path
      for path in nx.all_simple_paths(self.graph, x, y)
      if len(path) > 2
      and path[1] in self.dag.predecessors(x)
    ]

  def is_valid_backdoor_adjustment_set(self, x, y, z):
    """
    Test whether z is a valid backdoor adjustment set for
    estimating the causal impact of x on y via the backdoor
    adjustment formula:

    P(y|do(x)) = \sum_{z}P(y|x,z)P(z)

    Arguments
    ---------
    x: str
      Intervention Variable

    y: str
      Target Variable

    z: str or set[str]
      Adjustment variables

    Returns
    -------
    is_valid_adjustment_set: bool
    """
    z = _variable_or_iterable_to_set(z)

    assert x in self.observed_vars
    assert y in self.observed_vars
    assert x not in z
    assert y not in z

    if any([zz in nx.descendants(self.dag, x) for zz in z]):
      return False

    unblocked_backdoor_paths = [
      path
      for path in self.get_all_backdoor_paths(x, y)
      if not self._check_d_separation(path, z)
    ]

    if unblocked_backdoor_paths:
      return False

    return True

  def get_all_backdoor_adjustment_sets(self, x, y):
    """
    Get all sets of variables which are valid adjustment sets for
    estimating the causal impact of x on y via the back door 
    adjustment formula:

    P(y|do(x)) = \sum_{z}P(y|x,z)P(z)

    Note that the empty set can be a valid adjustment set for some CGMs,
    in this case frozenset(frozenset(), ...) is returned. This is different
    from the case where there are no valid adjustment sets where the
    empty set is returned.

    Arguments
    ---------
    x: str 
      Intervention Variable 
    y: str
      Target Variable

    Returns
    -------
    condition set: frozenset[frozenset[variables]]
    """
    assert x in self.observed_vars
    assert y in self.observed_vars

    possible_adjustment_variables = (
      set(self.observed_vars)
      - {x} - {y}
      - set(nx.descendants(self.dag, x))
    )

    valid_adjustment_sets = frozenset([
      frozenset(s)
      for s in _powerset(possible_adjustment_variables)
      if self.is_valid_backdoor_adjustment_set(x, y, s)
    ])

    return valid_adjustment_sets

  def is_valid_frontdoor_adjustment_set(self, x, y, z):
    """
    Test whether z is a valid frontdoor adjustment set for
    estimating the causal impact of x on y via the frontdoor
    adjustment formula:

    P(y|do(x)) = \sum_{z}P(z|x)\sum_{x'}P(y|x',z)P(x')

    Arguments
    ---------
    x: str
      Intervention Variable

    y: str
      Target Variable

    z: set
      Adjustment variables

    Returns
    -------
    is_valid_adjustment_set: bool
    """
    z = _variable_or_iterable_to_set(z)

    # 1. does z block all directed paths from x to y?
    unblocked_directed_paths = [
      path for path in
      nx.all_simple_paths(self.dag, x, y)
      if not any(zz in path for zz in z)
    ]

    if unblocked_directed_paths:
      return False

    # 2. no unblocked backdoor paths between x and z
    unblocked_backdoor_paths_x_z = [
      path
      for zz in z
      for path in self.get_all_backdoor_paths(x, zz)
      if not self._check_d_separation(path, z - {zz})
    ]

    if unblocked_backdoor_paths_x_z:
      return False

    # 3. x is a valid backdoor adjustment set for z
    if not all(self.is_valid_backdoor_adjustment_set(zz, y, x) for zz in z):
      return False

    return True

  def get_all_frontdoor_adjustment_sets(self, x, y):
    """
    Get all sets of variables which are valid frontdoor adjustment sets for
    estimating the causal impact of x on y via the frontdoor adjustment
    formula:

    P(y|do(x)) = \sum_{z}P(z|x)\sum_{x'}P(y|x',z)P(x')

    Note that the empty set can be a valid adjustment set for some CGMs,
    in this case frozenset(frozenset(), ...) is returned. This is different
    from the case where there are no valid adjustment sets where the
    empty set is returned.

    Arguments
    ---------
    x: str
      Intervention Variable
    y: str
      Target Variable

    Returns
    -------
    condition set: frozenset[frozenset[variables]]
    """
    assert x in self.observed_vars
    assert y in self.observed_vars

    possible_adjustment_variables = (
      set(self.observed_vars)
      - {x} - {y}
    )

    valid_adjustment_sets = frozenset(
      [
        frozenset(s)
        for s in _powerset(possible_adjustment_variables)
        if self.is_valid_frontdoor_adjustment_set(x, y, s)
      ])

    return valid_adjustment_sets
  
  def is_directly_transportable(self, y, zs=[]):
    if len(self.s_nodes) == 0:
      return True
    do_set_nodes = self.do_set_nodes()
    for s_node in self.s_nodes:
      if not do_set_nodes.is_d_separated(s_node, y, zs + list(self.set_nodes)):
        return False
    return True
  
  def get_adjustment_sets(self, x, y, zs):
    if len(zs) > 0:
      return self.get_all_backdoor_adjustment_sets(x, y)
    return frozenset(sorted(list(self.get_all_backdoor_adjustment_sets(x,y)) + list(self.get_all_frontdoor_adjustment_sets(x,y))))
  
  def is_trivially_transportable(self, x, y, zs=set()):
    for s in self.get_adjustment_sets(x, y, zs):
      if zs.issubset(s):
        return True
    return False
  
  def shortest_triv_transp_adj_set(self, x, y, zs=set()):
    shortest = None
    shortest_set_size = float('inf')
    for s in self.get_adjustment_sets(x, y, zs):
        if zs.issubset(s) and len(s) < shortest_set_size:
          shortest = s
          shortest_set_size = len(s)
    return shortest
  
  def triv_transp_adj_formula(self, x, y, zs=set()):
    ss = [v for v in self.shortest_triv_transp_adj_set(x,y,zs) if v not in zs]
    prob_query = [[{y: None}, {x: None}]]
    if len(ss) > 0:
      prob_query.append([dict(), dict()])
      for z in zs:
        prob_query[1][1][z] = None
      for s in ss:
        prob_query[0][1][s] = None
        prob_query[1][0][s] = None
    for z in zs:
      prob_query[0][1][z] = None
    return prob_query
  
  def do_set_nodes(self):
    new_model = self.__copy__()
    for node in self.set_nodes:
      new_model = new_model.do(node)
    return new_model
      
  def __copy__(self):
    return CausalGraph(
        nodes=self.get_non_s_nodes(), edges=self.get_edges(),
        latent_edges=self.get_latent_edges(), set_nodes=self.set_nodes,
        s_node_children=self.get_s_node_children())
    

def _variable_or_iterable_to_set(x):
  """
  Convert variable or iterable x to a frozenset.

  If x is None, returns the empty set.

  Arguments
  ---------
  x: None, str or Iterable[str]

  Returns
  -------
  x: frozenset[str]

  """
  if x is None:
    return frozenset([])

  if isinstance(x, str):
    return frozenset([x])

  if not isinstance(x, Iterable) or not all(isinstance(xx, str) for xx in x):
    raise ValueError(
      "{} is expected to be either a string or an iterable of strings"
      .format(x))

  return frozenset(x)

def _powerset(iterable):
  """
  https://docs.python.org/3/library/itertools.html#recipes
  powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
  """
  s = list(iterable)
  return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

if __name__ == "__main__":
  nodes = ["W", "X", "Y", "Z", "C"]
  edges = [("W","X"),("W", "Y"), ("X", "Z"), ("Z", "Y"), ("C", "Y")]
  model = CausalGraph(nodes, edges, set_nodes=["X"])
  bias_model = model.selection_diagram(["C"])
  # print(bias_model.is_directly_transportable("Y", ["W"]))
  print(bias_model.get_all_backdoor_adjustment_sets("X", "Y"))
  # print(bias_model.get_all_frontdoor_adjustment_sets("X", "Y"))
  print(bias_model.is_directly_transportable("Y", ["W"]))
  print(bias_model.is_trivially_transportable("X", "Y", {"W"}))
  print(bias_model.triv_transp_adj_formula("X", "Y", {"W"}))
