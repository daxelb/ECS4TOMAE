import gutil
import pandas as pd
import multiprocessing as mp
from copy import copy

class Sim:
  def __init__(self, world, num_episodes, num_trials):
    self.world = world
    self.num_episodes = num_episodes
    self.num_trials = num_trials
    self.trials = []
  
  def run(self):
    for i in range(self.num_trials):
      world = copy(self.world)
      for j in range(self.num_episodes):
        world.run_once()
        gutil.printProgressBar(
          i+((j+1)/self.num_episodes),
          self.num_trials,
          "{}:{}".format(i+1,j+1),
          )
      self.trials.append(world.pseudo_cum_regret)
    return

  def get_data(self):
    data = pd.DataFrame(columns=range(self.num_episodes))
    for trial in self.trials:
      if self.world.is_community:
        data.loc[len(data)] = trial
        continue
      for agent in self.world.agents:
        data.loc[len(data)] = gutil.list_from_dicts(trial, agent)
    return data
  
  def simulate(self, results, index):
    sim = copy(self)
    sim.run()
    results[index] = sim.get_data()

  def multithreaded_sim(self):
    num_threads = mp.cpu_count()
    jobs = []
    results = mp.Manager().list([None] * num_threads)
    for i in range(num_threads):
      job = mp.Process(target=self.simulate,args=(results, i))
      jobs.append(job)
      job.start()
    [job.join() for job in jobs]
    combined_results = pd.DataFrame(columns=range(self.num_episodes))
    for result in results:
      combined_results = pd.concat((combined_results, result), ignore_index=True)
    return combined_results

  def __copy__(self):
    return Sim(copy(self.world), self.num_episodes, self.num_trials)