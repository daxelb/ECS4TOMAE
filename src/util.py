def alphabetize(l):
  l.sort()
  return l

def num_combinations(list_of_lists):
  if not len(list_of_lists):
    return 0
  count = 1
  for lst in list_of_lists:
    count *= len(lst)
  return count

def combinations(dictionary):
  combos = list()
  num_combos = num_combinations(dictionary.values())

  for _ in range(num_combos):
    combos.append([None] * len(dictionary.keys()))

  for index, item in enumerate(dictionary.items()):
    var = item[0]
    domain = item[1]
    num_combos /= len(domain)
    count = 0
    pos_val_index = 0
    for combo in combos:
      if count >= num_combos:
        count = 0
        pos_val_index = pos_val_index + 1 if pos_val_index + 1 < len(domain) else 0
      combo[index] = (var, domain[pos_val_index])
      count += 1
  return combos

def conditional_prob(Q, e, dataset):
  Q_and_e = {**Q, **e}
  return prob(Q_and_e, dataset) / prob(e, dataset)

def prob(Q, dataset):
  total = len(list(dataset.values())[0])
  count = 0
  for i in range(len(list(dataset.values())[0])):
    consistent = True
    for key in Q:
      if Q[key] and dataset[key][i] != Q[key]:
        consistent = False
        break
    count += consistent
  return count / total

def parse_prob_query(str_query):
  parsed = []
  new_p = [{}, {}]
  query = True
  assignment = False
  Q_var = Q_val = e_var = e_val = ""
  for char in str_query:
    if char == 'P' or char == '(':
      query = True
      assignment = False
    elif char == '|':
      assignment = False
      Q_val = None if Q_val == "" else float(Q_val) if int(float(Q_val)) != float(Q_val) else int(float(Q_val))
      new_p[0][Q_var] = Q_val
      query = False
    elif char == ')':
      if query:
        Q_val = None if Q_val == "" else float(Q_val) if int(float(Q_val)) != float(Q_val) else int(float(Q_val))
        new_p[0][Q_var] = Q_val
      elif len(e_var):
        e_val = None if e_val == "" else float(e_val) if int(float(e_val)) != float(e_val) else int(float(e_val))
        new_p[1][e_var] = e_val
      parsed.append(tuple(new_p))
      Q_var = Q_val = e_var = e_val = ""
      new_p = [{}, {}]
    elif char == ',':
      assignment = False
      if query:
        Q_val = None if Q_val == "" else float(Q_val) if int(float(Q_val)) != float(Q_val) else int(float(Q_val))
        new_p[0][Q_var] = Q_val
        Q_var = Q_val = e_var = e_val = ""
      else:
        e_val = None if e_val == "" else float(e_val) if int(float(e_val)) != float(e_val) else int(float(e_val))
        new_p[1][e_var] = e_val
        Q_var = Q_val = e_var = e_val = ""
    elif char == '=':
      assignment = True
    else:
      if query:
        if assignment:
          Q_val += char
        else:
          Q_var += char
      else:
        if assignment:
          e_val += char
        else:
          e_var += char
  return parsed