from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np


Node = str
Edge = Tuple[Node, Node]


@dataclass(frozen=True)
class WalkResult:
    path: List[Node]
    visit_counts: Dict[Node, int]
    visit_distribution: Dict[Node, float]


@dataclass(frozen=True)
class PageRankRun:
    scores: Dict[Node, float]
    history: List[float]
    iterations: int


@dataclass(frozen=True)
class RLResult:
    q_values: Dict[Tuple[Node, Node], float]
    policy: Dict[Node, Node | None]
    episode_rewards: List[float]


def build_graph(nodes: Iterable[Node], edges: Iterable[Edge]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(str(node).strip() for node in nodes if str(node).strip())
    graph.add_edges_from(
        (str(source).strip(), str(target).strip())
        for source, target in edges
        if str(source).strip() and str(target).strip()
    )
    return graph


def parse_edges(edge_text: str) -> List[Edge]:
    edges: List[Edge] = []
    for raw_line in edge_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" not in line:
            raise ValueError(f"Use 'source -> target' format: {raw_line}")
        source, target = [part.strip() for part in line.split("->", 1)]
        if not source or not target:
            raise ValueError(f"Both source and target are required: {raw_line}")
        edges.append((source, target))
    return edges


def sorted_nodes(graph: nx.DiGraph) -> List[Node]:
    return sorted(str(node) for node in graph.nodes())


def transition_matrix(graph: nx.DiGraph, nodes: Sequence[Node] | None = None) -> np.ndarray:
    ordered = list(nodes or sorted_nodes(graph))
    n = len(ordered)
    matrix = np.zeros((n, n), dtype=float)
    index = {node: idx for idx, node in enumerate(ordered)}

    for source in ordered:
        successors = sorted(str(node) for node in graph.successors(source))
        if not successors:
            continue
        probability = 1.0 / len(successors)
        for target in successors:
            matrix[index[source], index[target]] = probability
    return matrix


def dangling_nodes(graph: nx.DiGraph) -> List[Node]:
    return [node for node in sorted_nodes(graph) if graph.out_degree(node) == 0]


def communication_classes(graph: nx.DiGraph) -> List[Dict[str, object]]:
    classes = []
    for component in nx.strongly_connected_components(graph):
        members = sorted(str(node) for node in component)
        closed = True
        for node in members:
            for target in graph.successors(node):
                if target not in component:
                    closed = False
                    break
            if not closed:
                break
        classes.append({"nodes": members, "closed": closed})
    return sorted(classes, key=lambda item: (not bool(item["closed"]), item["nodes"]))


def absorbing_states(graph: nx.DiGraph) -> List[Node]:
    absorbing = []
    for node in sorted_nodes(graph):
        successors = set(graph.successors(node))
        if successors == {node}:
            absorbing.append(node)
    return absorbing


def stationary_distribution(
    graph: nx.DiGraph,
    max_iter: int = 10_000,
    tolerance: float = 1e-12,
) -> Tuple[Dict[Node, float], List[float]]:
    nodes = sorted_nodes(graph)
    n = len(nodes)
    if n == 0:
        return {}, []

    matrix = transition_matrix(graph, nodes)
    # Treat dangling rows as uniform jumps so the Markov chain remains valid.
    for row in range(n):
        if np.isclose(matrix[row].sum(), 0.0):
            matrix[row] = np.ones(n) / n

    distribution = np.ones(n) / n
    history: List[float] = []
    for _ in range(max_iter):
        new_distribution = distribution @ matrix
        delta = float(np.linalg.norm(new_distribution - distribution, ord=1))
        history.append(delta)
        distribution = new_distribution
        if delta < tolerance:
            break

    return dict(zip(nodes, distribution.round(8))), history


def run_random_walk(
    graph: nx.DiGraph,
    start: Node,
    steps: int,
    seed: int | None = None,
) -> WalkResult:
    if start not in graph:
        raise ValueError("Start node is not in the graph.")
    rng = np.random.default_rng(seed)
    current = start
    path = [current]

    for _ in range(steps):
        successors = sorted(str(node) for node in graph.successors(current))
        if not successors:
            successors = sorted_nodes(graph)
        current = str(rng.choice(successors))
        path.append(current)

    counts = {node: path.count(node) for node in sorted_nodes(graph)}
    total = len(path)
    distribution = {node: count / total for node, count in counts.items()}
    return WalkResult(path=path, visit_counts=counts, visit_distribution=distribution)


def pagerank_power_iteration(
    graph: nx.DiGraph,
    damping: float = 0.85,
    max_iter: int = 200,
    tolerance: float = 1e-10,
) -> PageRankRun:
    nodes = sorted_nodes(graph)
    n = len(nodes)
    if n == 0:
        return PageRankRun({}, [], 0)

    matrix = transition_matrix(graph, nodes)
    for row in range(n):
        if np.isclose(matrix[row].sum(), 0.0):
            matrix[row] = np.ones(n) / n

    google_matrix = damping * matrix + (1.0 - damping) * (np.ones((n, n)) / n)
    rank = np.ones(n) / n
    history: List[float] = []
    for iteration in range(1, max_iter + 1):
        new_rank = rank @ google_matrix
        error = float(np.linalg.norm(new_rank - rank, ord=1))
        history.append(error)
        rank = new_rank
        if error < tolerance:
            break

    return PageRankRun(dict(zip(nodes, rank.round(8))), history, len(history))


def compare_damping_factors(
    graph: nx.DiGraph,
    factors: Sequence[float] = (0.6, 0.75, 0.85, 0.95),
) -> Dict[float, PageRankRun]:
    return {factor: pagerank_power_iteration(graph, factor) for factor in factors}


def value_iteration_shortest_policy(
    graph: nx.DiGraph,
    goal: Node,
    gamma: float = 0.9,
    iterations: int = 200,
) -> Dict[Node, Node | None]:
    nodes = sorted_nodes(graph)
    values = {node: 0.0 for node in nodes}
    policy: Dict[Node, Node | None] = {node: None for node in nodes}

    for _ in range(iterations):
        new_values = values.copy()
        for node in nodes:
            if node == goal:
                new_values[node] = 10.0
                policy[node] = None
                continue
            actions = sorted(str(successor) for successor in graph.successors(node))
            if not actions:
                new_values[node] = -5.0
                policy[node] = None
                continue
            action_values = {
                action: (-1.0 + (10.0 if action == goal else gamma * values[action]))
                for action in actions
            }
            best_action = max(action_values, key=action_values.get)
            new_values[node] = action_values[best_action]
            policy[node] = best_action
        values = new_values

    return policy


def q_learning(
    graph: nx.DiGraph,
    start: Node,
    goal: Node,
    episodes: int = 300,
    max_steps: int = 30,
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    seed: int | None = None,
) -> RLResult:
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must both be in the graph.")

    rng = np.random.default_rng(seed)
    q_values: Dict[Tuple[Node, Node], float] = {}
    for node in sorted_nodes(graph):
        for target in graph.successors(node):
            q_values[(node, str(target))] = 0.0

    rewards: List[float] = []
    for _ in range(episodes):
        state = start
        total_reward = 0.0
        for _ in range(max_steps):
            actions = sorted(str(target) for target in graph.successors(state))
            if not actions:
                total_reward -= 5.0
                break
            if rng.random() < epsilon:
                action = str(rng.choice(actions))
            else:
                action = max(actions, key=lambda candidate: q_values.get((state, candidate), 0.0))

            reward = 10.0 if action == goal else -1.0
            future_actions = sorted(str(target) for target in graph.successors(action))
            best_future = max((q_values.get((action, nxt), 0.0) for nxt in future_actions), default=0.0)
            old_value = q_values.get((state, action), 0.0)
            q_values[(state, action)] = old_value + alpha * (reward + gamma * best_future - old_value)

            total_reward += reward
            state = action
            if state == goal:
                break
        rewards.append(total_reward)

    policy: Dict[Node, Node | None] = {}
    for node in sorted_nodes(graph):
        actions = sorted(str(target) for target in graph.successors(node))
        policy[node] = max(actions, key=lambda candidate: q_values.get((node, candidate), 0.0)) if actions else None

    return RLResult(q_values=q_values, policy=policy, episode_rewards=rewards)


def ranked_items(scores: Dict[Node, float]) -> List[Tuple[Node, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
