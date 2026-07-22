# Interactive PageRank and Random Walk Analysis Toolkit

This is my SOS project for exploring how directed graphs behave under Markov chains, random walks, PageRank, and a small reinforcement learning extension. The main idea is simple: build one graph, then use it in different ways to understand movement, ranking, decision making, and learning.

## Features

- Build your own directed graph from an edge list.
- Visualize the graph with NetworkX and Matplotlib.
- Compute the transition matrix.
- Show communication classes using strongly connected components.
- Detect absorbing states and dangling nodes.
- Compute a stationary distribution.
- Run random walk simulations and plot empirical visit frequencies.
- Compute PageRank with power iteration.
- Compare damping factors `0.6`, `0.75`, `0.85`, and `0.95`.
- Plot PageRank convergence curves.
- Show node rankings.
- Extend the graph into a travel problem using MDP value iteration and Q-learning.

## Project Story

The report can naturally flow like this:

```text
Markov Chain
  -> Random Walk
  -> PageRank
  -> Decision Making with MDP
  -> Learning with Reinforcement Learning
```

In a normal Markov chain, the next node is chosen randomly from outgoing edges. In an MDP, an agent chooses an action, such as which outgoing edge to take. In reinforcement learning, the agent learns those choices from rewards.

I kept the extension small on purpose. It supports the main topic instead of turning the project into a separate RL project.

## Setup

Install Python 3.11 or newer, then run these commands in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If your system uses the Python launcher instead of `python`, use:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
streamlit run app.py
```

After the command starts, Streamlit will show a local URL, usually `http://localhost:8501`.

## How to Use

1. Choose a sample graph or type edges in the sidebar using `A -> B` format.
2. Open the Markov Chain tab to inspect the transition matrix and classes.
3. Open the Random Walk tab to simulate movement through the graph.
4. Open the PageRank tab to compare damping factors and rankings.
5. Open the MDP and RL tab to pick a start and goal for the traveler.

## Example Edge List

```text
Home -> Cafe
Home -> Library
Cafe -> Station
Library -> Station
Library -> Park
Park -> Museum
Station -> Museum
Museum -> Goal
Cafe -> Park
```

## Core Methods

The code uses a row-stochastic transition matrix:

```text
P[i, j] = probability of moving from node i to node j
```

Stationary distribution is found by repeatedly multiplying:

```text
pi_next = pi * P
```

PageRank adds damping and teleportation:

```text
G = dP + (1 - d) / N
```

The RL extension uses Q-learning with reward `+10` for reaching the goal, `-1` for each normal move, and `-5` for dead ends.

## Tests

Run:

```powershell
pytest
```

The tests cover transition matrices, communication classes, absorbing states, stationary distributions, random walks, PageRank, and Q-learning policy output.

## Files

- `app.py` contains the interactive Streamlit app.
- `src/markov_toolkit.py` contains the actual graph, Markov chain, PageRank, MDP, and RL logic.
- `tests/test_markov_toolkit.py` contains small correctness tests.
- `samples/` contains graph examples that can be pasted into the app.
- `report_outline.md` gives a ready structure for the written SOS report.
- `project_notes.md` has demo notes and explanation points.
