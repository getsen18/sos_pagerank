import numpy as np

from src.markov_toolkit import (
    absorbing_states,
    build_graph,
    communication_classes,
    compare_damping_factors,
    pagerank_power_iteration,
    parse_edges,
    q_learning,
    run_random_walk,
    stationary_distribution,
    transition_matrix,
)


def test_transition_matrix_is_row_stochastic_for_non_dangling_nodes():
    graph = build_graph(["A", "B", "C"], [("A", "B"), ("A", "C"), ("B", "C")])
    matrix = transition_matrix(graph, ["A", "B", "C"])

    assert np.allclose(matrix[0], [0.0, 0.5, 0.5])
    assert np.allclose(matrix[1], [0.0, 0.0, 1.0])
    assert np.allclose(matrix[2], [0.0, 0.0, 0.0])


def test_communication_classes_and_absorbing_state():
    graph = build_graph(["A", "B", "C"], [("A", "B"), ("B", "A"), ("B", "C"), ("C", "C")])
    classes = communication_classes(graph)

    assert {"nodes": ["C"], "closed": True} in classes
    assert absorbing_states(graph) == ["C"]


def test_stationary_distribution_sums_to_one():
    graph = build_graph(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])
    distribution, _ = stationary_distribution(graph)

    assert abs(sum(distribution.values()) - 1.0) < 1e-6


def test_random_walk_records_every_step():
    graph = build_graph(["A", "B"], [("A", "B"), ("B", "A")])
    walk = run_random_walk(graph, "A", 10, seed=1)

    assert len(walk.path) == 11
    assert abs(sum(walk.visit_distribution.values()) - 1.0) < 1e-9


def test_pagerank_and_damping_comparison():
    graph = build_graph(["A", "B", "C"], parse_edges("A -> B\nB -> C\nC -> A"))
    run = pagerank_power_iteration(graph, damping=0.85)
    comparison = compare_damping_factors(graph)

    assert set(comparison) == {0.6, 0.75, 0.85, 0.95}
    assert abs(sum(run.scores.values()) - 1.0) < 1e-6


def test_q_learning_returns_policy():
    graph = build_graph(["A", "B", "Goal"], [("A", "B"), ("B", "Goal")])
    result = q_learning(graph, "A", "Goal", episodes=50, seed=2)

    assert result.policy["A"] == "B"
    assert result.policy["B"] == "Goal"
