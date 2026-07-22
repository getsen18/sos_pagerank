# Report Outline

## Title

Interactive Analysis of Markov Chains, PageRank, and Learning on Directed Graphs

## 1. Motivation

Many real systems can be represented as directed graphs: webpages linking to each other, routes between places, or states in a process. The aim of this project is to make these ideas easier to understand by letting the user build a graph and immediately observe the mathematical behavior behind it.

## 2. Markov Chain Model

A directed graph is converted into a transition matrix. If a node has outgoing edges, each outgoing edge is treated as equally likely. This creates a simple Markov chain where the next state depends only on the current state.

Important points to discuss:

- The rows of the transition matrix represent current nodes.
- The columns represent possible next nodes.
- Communication classes are found using strongly connected components.
- Absorbing states are nodes that transition only to themselves.
- Dangling nodes are nodes with no outgoing edges.

## 3. Random Walk

A random walk simulates a person or surfer moving through the graph by repeatedly choosing one outgoing edge at random. After many steps, the visit frequency gives an experimental idea of long-run behavior.

This section is useful because it connects the matrix theory to something visible and intuitive.

## 4. Stationary Distribution

The stationary distribution describes the long-run probability of being at each node. In the project, it is computed iteratively by repeatedly multiplying the distribution vector by the transition matrix.

## 5. PageRank

PageRank modifies the random walk model by adding a damping factor. With probability `d`, the surfer follows a link. With probability `1 - d`, the surfer teleports to a random node. This prevents the ranking from getting trapped in dead ends or isolated parts of the graph.

The toolkit compares damping factors `0.6`, `0.75`, `0.85`, and `0.95`. The convergence graph shows how quickly each setting settles.

## 6. Node Ranking

Nodes are ranked by their final PageRank score. This gives a practical interpretation of the math: more important nodes receive higher scores because they are reached more often or linked from important nodes.

## 7. Extension: From Random Movement to Decisions

The project includes a small travel example to connect Markov chains with decision making.

- In a Markov chain, the traveler moves randomly.
- In an MDP, the traveler chooses among outgoing edges.
- In reinforcement learning, the traveler learns good choices using rewards.

The app uses value iteration for the MDP policy and Q-learning for the learned policy.

## 8. Learning Setup

The Q-learning extension uses:

- `+10` reward for reaching the goal.
- `-1` reward for every normal move.
- `-5` penalty for dead ends.

The reward curve helps show whether the agent improves over repeated episodes.

## 9. Results to Include

Suggested screenshots:

- A custom graph drawn in the app.
- Transition matrix and communication classes.
- Random walk visit distribution.
- PageRank convergence plot for different damping factors.
- Node ranking table.
- MDP/RL policy graph for a travel problem.

## 10. Limitations

The current version assumes equal probability on outgoing edges. A useful future improvement would be weighted edges, where some links or routes are more likely than others. Another extension would be exporting graph results as a CSV or PDF.

## 11. Conclusion

This project shows how one directed graph can support a complete learning path: Markov chains explain random movement, random walks motivate PageRank, PageRank ranks nodes, and MDP/RL ideas extend the same graph into decision making and learning.
