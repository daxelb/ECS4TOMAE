from query import Query

def num_consistent(data, query):
  """
  Returns the number of datapoints in the dataset
  that are consistent with a probability query.
  """
  return sum([all([query[var] == dp[var] for var in query]) for dp in data])

def uncond_prob(data, query):
  """
  Calculates an probability query that excludes
  evidence/conditional arguments that would follow
  a conditioning bar.
  
  Ex:
  P(A,B) = P(A ∩ B)
    => 0.5
  """
  return num_consistent(data, query) / len(data) if query else 1.0

def prob(data, query):
  """
  For a query for which all queried-on variables
  are assigned, returns the conditional probability 
  calculated from the dataset.
  
  query: a Query object.
  """
  assert query.all_assigned()
  uncomputed = uncomputed_prob(data, query)
  return None if uncomputed[1] == 0 else uncomputed[0] / uncomputed[1]

def uncomputed_prob(data, query):
  assert query.all_assigned()
  return (num_consistent(data, Query(query.Q_and_e())), num_consistent(data, Query(query.e)))