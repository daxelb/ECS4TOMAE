import copy
import util

def prob(dataset, Q, e={}):
  """
  For a query for which all queried-on variables
  are assigned, returns the conditional probability 
  calculated from the dataset.
  """
  assert not (get_unassigned(Q) or get_unassigned(e))
  return uncond_prob(dataset, {**Q, **e}) / uncond_prob(dataset, e)

def uncond_prob(dataset, Q):
  """
  Calculates an probability query that excludes
  evidence/conditional arguments that would follow
  a conditioning bar.
  
  Ex:
  P(A,B) = P(A ∩ B)
    => 0.5
  """
  if not Q:
    return 1.0
  total = len(dataset)
  count = 0
  for i in range(total):
    consistent = True
    for key in Q:
      if Q[key] != None and dataset[i][key] != Q[key]:
        consistent = False
        break
    count += consistent
  return count / total

def over(domains, node, queries):
  new_queries = []
  node_domain = domains[node]
  for n_ass in node_domain:
    for query in queries:
      new_query = copy.deepcopy(list(query))
      for e in new_query:
        if node in e:
          e[node] = n_ass
      new_queries.append(tuple(new_query))
  return new_queries

def summation(data, queries):
  summation = 0
  for query in queries:
    summation += prob(data, query[0], query[1])
  return summation

def product(data, queries):
  product = 1
  for query in queries:
    product *= prob(data, query[0], query[1])
  return product
    
def prob_from_cpts(data, model, query):
  eqn = [[],[]]
  num = {**query[0], **query[1]}
  for node in num:
    for pa in model.pa(node):
      cpt = model.get_node_dist(pa)
      for e in cpt:
        for key in e:
          if key in num:
            e[key] = num[key]
      cpt = tuple(cpt)
      if node in query[1]:
        eqn[1].append(cpt)
      eqn[0].append(cpt)
  [util.remove_dup_queries(e) for e in eqn]
  return summation(data, product(over_unassigned(eqn[0]))) / summation(data, product(over_unassigned(eqn[1])))

def get_unassigned(queries):
  unassigned = set()
  for query in queries:
    for e in query:
      for key in e:
        if e[key] is None:
          unassigned.add(key)
  return unassigned

def over_unassigned(domains, queries):
  unassigned = get_unassigned(queries)
  if not unassigned: return queries
  for u in unassigned:
    return over_unassigned(over(domains, u, queries))
  
