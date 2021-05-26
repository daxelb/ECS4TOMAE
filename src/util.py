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

def get_combinations(dictionary):
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