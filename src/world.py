import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from cam import CausalAssignmentModel, discrete_model
import numpy as np
import util

if __name__ == "__main__":
    domains = {"W": (0, 1), "X": (0, 1), "Z": (0, 1), "Y": (0, 1)}
    environment = Environment(domains, {
      "W": lambda: np.random.choice([0, 1], p=[0.5, 0.5]),
      "X": CausalAssignmentModel(["W"], None),
      "Z": discrete_model(["X"], {(0,): [0.75, 0.25], (1,): [0.25, 0.75]}),
      "Y": discrete_model(["W", "Z"], {(0, 0): [1, 0], (0, 1): [1, 0], (1, 0): [1, 0], (1, 1): [0, 1]})
    })
    agent0 = Agent("zero", environment, "Y")
    agent1 = Agent("one", environment, "Y")

    correct_div_nodes = {
        "zero": {"one": {"W": False, "Y": False, "Z": False}},
        "one": {"zero": {"W": False, "Y": False, "Z": False}}}

    iterations = []
    num_correct = {"zero": [], "one": []}
    for i in range(500):
        iterations.append(i)
        agent0.act()
        agent1.act()
        agent0.encounter(agent1)
        agent1.encounter(agent0)
        num_correct["zero"].append(util.num_matches(
            agent0.divergent_nodes()["one"], correct_div_nodes["zero"]["one"]))
        num_correct["one"].append(util.num_matches(
            agent1.divergent_nodes()["zero"], correct_div_nodes["one"]["zero"]))
    total_correct = [0] * len(list(num_correct.values())[0])
    for key in num_correct:
        for i, e in enumerate(num_correct[key]):
            total_correct[i] += e

    plt.plot(iterations, total_correct)
    plt.xlabel("Iterations")
    plt.ylabel("Num. correct IDs of S-node locations")
    plt.show()
        
