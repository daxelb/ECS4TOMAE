
#  Exploiting Causal Structure for Transportability in Online, Multi-Agent Environments (ECS4TOMAE)

Read the paper [here](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p199.pdf).

Source code for simulating and comparing agent peer-communication policies described in the titular paper.

## Setup

[!] Use [Python 3.9](https://www.python.org/downloads/)

1. Clone the repository

2. Navigate to the destination folder in a terminal window (using `cd`)

3. Download and install project dependencies: `pip install -r "requirements.txt"`

## Running Experiments / Simulations

- Modify parameters under `if __name__ == "__main__":` at the bottom of `sim.py`
	- To run an experiment (comparing cpr and poa across multiple inputs) input a tuple/list containing the different values of independent variables in the field.

	- To reproduce a previous simulation:
		- Find its `values.json` file found in the result's directory.

		- Use the values from this json file as parameters in `sim.py`
	

- Navigate to the `src` folder in the terminal

- To run the sim/experiment use `python sim.py`

### Sim Parameters
- `environment_dicts`
	- required even if environments are randomized since this also defines causal structure and the number of agents in the world
- `otp`
	- one of, or a combination of `OTP.SOLO`, `OTP.NAIVE`, `OTP.SENSITIVE`, `OTP.ADJUST`
- `asr`
	- one of, or a combination of `ASR.EG` `ASR.EF`, `ASR.ED`, `ASR.TS`
	- to test different "communities", you can also define a 2D tuple/list:
		- e.g. `((ASR.TS, ASR.TS, ASR.TS, ASR.TS), (ASR.TS, ASR.TS, ASR.EF, ASR.EF), (ASR.EF, ASR.EF, ASR.EF, ASR.EF))`
	- You can also use `combinations_with_replacement` from `itertools` (imported) to automate a gradient between communities
		- e.g. `combinations_with_replacement((ASR.TS, ASR.EF), 4)`
- `T`
	- The number of total trials in a simulation
- `mc_sims`
	- The number of times a Monte Carlo repetition of a sim is run on each process in the CPU.
	- `N = num_agents * mc_sims * num_processes`
- `tau`
	- Defines the threshold value for Hellinger Distance between two node probability distributions for which data will be accepted/rejected
- `EG_epsilon`
	- Value of epsilon used in Epsilon Greedy ASR
- `EF_rand_trials`
	- The number of randomized/experimental trials per context for the Epsilon First ASR
- `ED_cooling_rate`
	- The geometric cooling/decay rate of epsilon (starting at 1.0) for the Epsilon Decreasing ASR
- `is_community`
	- A boolean value which is true when agent-assignments are standardized per world, false if randomized
- `rand_envs`
	- A boolean which is true when probability distributions in the SCM of an Environment are randomized, false if this probability distribution is predetermined in `environment_dicts`
- `node_mutation_chance`
	- A continuous value 0 <= _nmc_ <= 1 which determines the probability of a node's probability distribution being scrambled in a generated SCM if `rand_envs=True`
- `show`
	- If True, will display the result plot at the end of sim
-`save`
	- If True, will save results, values, and plots in a directory in `../output`
- `seed`
	- If None, will generate a random seed for which random numbers are generated from during simulation
	- If a positive integer is passed, this seed will dictate the randomization of the sim, meaning data is reproduceable if this seed is used again

## Code Overview
### `agent.py`
Defines agent classes and their behaviors.
- `Agent` - a superclass for the following classes
- `SoloAgent`
- `NaiveAgent`
- `SensitiveAgent`
- `AdjustAgent` 
### `assignment_models.py`
Defines assignment models, which describe the behavior of certain nodes in an SCM. These include:
- `ActionModel` - probability distribution is determined by an agent, and is not pre-defined
- `RandomModel` - an assignment model with no parents, probabilities are not based on other nodes
- `DiscreteModel` - an assignment with parents, whose outputs and their probabilities is based on input from parent node(s)
### `cgm.py`
Defines `CausalGraph` class and many helper methods that agents can access. This is a SCM without the probability distributions. Basically a directed, acyclic graph with methods that are helpful in causal inquiry.
### `cpt.py`
Defines the `CPT` class which is used to store node-specific data for agents to query.
### `enums.py`
Defines the `OTP` enum type and `ASR` enum type which determine agent behavior.
### `environment.py`
Defines the `Environment` class which interacts with agents to deliver contextual information and rewards to the agent (based on their action).
### `plot_xl.py`
Used to create and save a plotly object from an Excel (.xlsx) file. Modify variables at the top of this file to create plots.
### `process.py`
Defines per-process simulation methods. These objects return results to a `Sim` object, which combines and processes them.
### `query.py`
Defines classes (listed below) that represent probability queries and "count" queries. These objects can be used in combination with `CPT` objects to solve probability queries on node-specific data.
- `Query` - P(Y|X,W)
- `Count` - child class of query. Looks like: N(Y,X,W)
- `Queries` - parent class of Summation and Product. Looks like: [P(Y|Z,W), P(W|X)]
- `Summation` - P(Y|Z,W) + P(W|X) 
- `Product` - P(Y|Z,W) * P(W|X)
### `scm.py`
Defines the `SCM` class
### `sim.py`
Defines the `Sim` class, which is as the base-file for running simulations with parameters.
### `util.py`
A file containing several methods that are useful throughout the code-base.
### `world.py`
Defines the `World` class which agents (and their environments) interact within, communicating, acting, etc. This class also stored Cumulative Pseudo Regret and Probability of Optimal Action for the agents, which is returned and combined with other `World` results during a sim.
