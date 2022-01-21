"""
Defines the Causal Graphical Model class which tracks the nodes and edges
of a Structural Causal Model like a Bayesian Network. Does not store "Assignment Models"
i.e. probability distributions.

CREDIT:
Much of this file/class was written by Iain Barr (#ijmbarr on GitHub)
from his public repository, causalgraphicalmodels, which is registered with the MIT License.
The code has been imported and modified into this project for ease/consistency
"""

import networkx as nx
import graphviz
from itertools import combinations, chain
from collections.abc import Iterable
from query import Query, Product


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

  def get_observable(self):
    return sorted(list(self.observed_vars))

  def get_parents(self, nodes):
    if isinstance(nodes, str):
      return set(self.dag.predecessors(nodes))
    return set.union(*[self.dag.predecessors(node) for node in nodes])

  def get_children(self, nodes):
    if isinstance(nodes, str):
      return set(self.dag.successors(nodes))
    return set.union(*[self.dag.successors(node) for node in nodes])

  def get_ancestors(self, nodes):
    if isinstance(nodes, str):
      return nx.ancestors(self.dag, nodes)
    return set.union(*[nx.ancestors(self.dag, node) for node in nodes])

  def get_descendants(self, nodes):
    if isinstance(nodes, str):
      return nx.descendants(self.dag, nodes)
    return set.union(*[nx.descendants(self.dag, node) for node in nodes])

  def get_exogenous(self):
    return {n for n in self.dag.nodes if not self.get_parents(n)}

  def get_endogenous(self):
    return {n for n in self.dag.nodes if self.get_parents(n)}

  def get_leaves(self):
    return {n for n in self.dag.nodes if not self.get_children(n)}

  def get_unset_nodes(self):
    return {n for n in self.dag.nodes if n not in self.set_nodes}

  def get_feat_vars(self, act_var):
    return self.cgm.get_parents(act_var)

  def causal_path(self, n1, n2):
    """
    Returns the nodes along the causal path from
    n1 to n2.
    Ex:
           Z
         /   \
        /     \
       v       v
      X -> W -> Y

      causal_path(X, Y) = {W, Y}
    """
    path_nodes = self.get_ancestors(n2).intersection(self.get_descendants(n1))
    path_nodes.add(n2)
    return path_nodes

  def draw_model(self, v=True):
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

  def get_dist(self, node=None):
    return self.get_model_dist() if node is None else self.get_node_dist(node)

  def get_model_dist(self):
    """
    Returns a the Markovian Factorization (MF) of the model's
    Joint Probability Distribution (JPD).
    """
    product = Product()
    for node in nx.topological_sort(self.dag):
      product.append(Query(node, self.get_parents(node)))
    return product

  def get_dist_as_dict(self, res):
    if isinstance(res, str):
      parents = sorted(list(self.get_parents(res)))
      return {res: self.get_dist_as_dict(parents)}
    elif isinstance(res, list):
      if len(res) == 1:
        res == res[0]
      elif len(res) == 0:
        return None
      new_dict = {}
      for node in res:
        parents = sorted(list(self.get_parents(node)))
        new_dict[node] = self.get_dist_as_dict(parents)
      return new_dict
    return res if res else None

  def get_node_dist(self, node):
    """
    Returns conditional probability distribution
    for a node in the model.
    Ex: A -> B <- C
    => <Query: P(B|A,C)>
    """
    return Query(node, self.get_parents(node))

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
    return set.union(*[self.get_children(n) for n in self.s_nodes])

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
    s_node_children = _variable_or_iterable_to_set(s_node_children)
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

  def is_directly_transportable(self, y, zs=set()):
    if len(self.s_nodes) == 0:
      return True
    do_set_nodes = self.do_set_nodes()
    for s_node in self.s_nodes:
      if not do_set_nodes.is_d_separated(s_node, y, set(list(zs) + list(self.set_nodes))):
        return False
    return True

  def get_adjustment_sets(self, x, y, zs):
    if len(zs) > 0:
      return self.get_all_backdoor_adjustment_sets(x, y)
    return set(list(self.get_all_backdoor_adjustment_sets(x, y)) + list(self.get_all_frontdoor_adjustment_sets(x, y)))

  def is_trivially_transportable(self, x, y, zs=set()):
    zs = _variable_or_iterable_to_set(zs)
    for s in self.get_adjustment_sets(x, y, zs):
      if zs.issubset(s) and s.issubset(set(self.get_non_s_nodes())):
        return True
    return False

  def shortest_triv_transp_adj_set(self, x, y, zs=set()):
    zs = set(zs)
    shortest = None
    shortest_set_size = float('inf')
    for s in self.get_adjustment_sets(x, y, zs):
      if zs.issubset(s) and len(s) < shortest_set_size and s.issubset(set(self.get_non_s_nodes())):
        shortest = s
        shortest_set_size = len(s)
    return shortest

  def direct_adj_formula(self, x, y, zs):
    return Query(y, [x] + list(zs))

  def trivial_adj_formula(self, x, y, zs):
    ss = [v for v in self.shortest_triv_transp_adj_set(x, y, zs) if v not in zs]
    prob_query = Product(Query(y, ss + list(zs) + [x]))  # [[{y: None}, {x: None}]]
    if len(ss) > 0:
      prob_query.append(Query(ss, zs))
    return prob_query

  def get_transport_formula(self, x, y, zs=set()):
    if self.is_directly_transportable(y, zs):
      return Product(self.direct_adj_formula(x, y, zs))
    if self.is_trivially_transportable(x, y, zs):
      return self.trivial_adj_formula(x, y, zs)
    return None

  def from_cpts(self, query):
    """
    Returns the input query in terms
    of the model's CPTs.
    """
    return Product([
        q for q in
        self.get_dist()
        if q.contains(query.g())
    ])

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
  nodes = ["W", "X", "Y", "Z"]
  edges = [("W", "X"), ("W", "Y"), ("X", "Z"), ("Z", "Y")]
  model = CausalGraph(nodes, edges, set_nodes={"X"})
  # print(model.get_all_backdoor_adjustment_sets("Y", "X", "Z"))
  # print(model.get_all_backdoor_paths("Y", "Z"))
  print(model.get_dist_as_dictionaries("Y"))
  # bias_model = model.selection_diagram({"Z"})
  # print(model.from_cpts(Query("Y", {"X", "W"})).assign({"X": 1}))
