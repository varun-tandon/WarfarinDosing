# Various Bandits in the Context of Warfarin Dosing 

## Setup

1. Create a virtualenv or conda environment
2. Run `pip install -r requirements.txt`
3. Prepare the dataset by running `python clean_dataset.py` to generate the version of the data that our models will run.

## Run agents

To run an agent use `python main.py --agent=AGENT_NAME` to generate a set of 20 experiments. Examples of agents are `fixed, linear, ucb, linucb, supervised-lin, supervised-ridge, thompson-0, ensemble-0`

## Generate plots 

Run `python plot.py --agent=AGENT_NAME1,AGENT_NAME2` to visualize that agent's performance against the
other agents provided. This automatically generates the named plots in the plots/ directory.