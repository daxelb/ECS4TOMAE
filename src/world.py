import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from assignment_models import AssignmentModel, discrete_model, random_model, action_model
import numpy as np
import random
import util

class World:
    def __init__(self, agents):
        self.agents = agents
        self.cdn = self.get_correct_div_nodes()
        self.num_correct = {}
        [self.num_correct.update({a.name: []}) for a in self.agents]
        self.total_correct = []
        return

    def act(self):
        [a.act() for a in self.agents]

    def encounter_all(self):
        for a in self.agents:
            friends = [f for f in self.agents if f != a]
            if not friends: return False
            [a.encounter(f) for f in friends]
    
    def encounter_one(self):
        for a in self.agents:
            friends = [f for f in self.agents if f != a]
            if not friends: return False
            a.encounter(random.choice(friends))
        return True
    
    def get_correct_div_nodes(self):
        correct_div_nodes = {}
        for a in self.agents:
            correct_div_nodes[a.name] = {}
            for f in self.agents:
                if a == f:
                    continue
                correct_div_nodes[a.name][f.name] = {}
                for node, assignment in a.environment._assignment.items():
                    correct_div_nodes[a.name][f.name][node] = assignment != f.environment._assignment[node]
        return correct_div_nodes

    def update_num_correct(self):
        total_correct = 0
        for a in self.agents:
            for f in self.agents:
                if a == f:
                    continue
                correct = util.num_matches(
                    a.divergent_nodes()[f.name], self.cdn[a.name][f.name])
                total_correct += correct
                self.num_correct[a.name].append(correct)
        self.total_correct.append(total_correct)



if __name__ == "__main__":
    baseline = {
        "W": random_model((0.5, 0.5)),
        "X": AssignmentModel(("W"), None, (0, 1)),
        "Z": discrete_model(("X"), {(0,): (0.75, 0.25), (1,): (0.25, 0.75)}),
        "Y": discrete_model(("W", "Z"), {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (1, 0), (1, 1): (0, 1)})
    }
    w1 = dict(baseline)
    w1["W"] = random_model((0.1, 0.9))
    w9 = dict(baseline)
    w9["W"] = random_model((0.9, 0.1))
    z5 = dict(baseline)
    z5["Z"] = discrete_model(("X"), {(0,): (0.8, 0.2), (1,): (0.5, 0.5)})

    agents = [
        Agent("zero", Environment(baseline), "Y"),
        Agent("one", Environment(baseline), "Y"),
        Agent("two", Environment(w1), "Y"),
        Agent("three", Environment(w9), "Y"),
        Agent("four", Environment(z5), "Y")
    ]

    world = World(agents)
    cdn = world.get_correct_div_nodes()

    r = 250
    for i in range(r):
        world.act()
        world.encounter_all()
        world.update_num_correct()

    plt.plot(np.arange(r), np.array(world.total_correct))
    plt.xlabel("Iterations")
    plt.ylabel("Num. correct IDs of S-node locations")
    plt.show()
        
