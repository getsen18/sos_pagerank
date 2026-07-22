from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

from src.markov_toolkit import (
    absorbing_states,
    build_graph,
    communication_classes,
    compare_damping_factors,
    dangling_nodes,
    pagerank_power_iteration,
    parse_edges,
    q_learning,
    ranked_items,
    run_random_walk,
    sorted_nodes,
    stationary_distribution,
    transition_matrix,
    value_iteration_shortest_policy,
)


SAMPLES = {
    "Mini web graph": """A -> B
A -> C
B -> C
C -> A
C -> D
D -> C
E -> D
E -> F
F -> E""",
    "Absorbing chain": """A -> B
B -> C
C -> C
D -> A
D -> C""",
    "Travel graph": """Home -> Cafe
Home -> Library
Cafe -> Station
Library -> Station
Library -> Park
Park -> Museum
Station -> Museum
Museum -> Goal
Cafe -> Park""",
}


def draw_graph(graph: nx.DiGraph, pagerank_scores=None, policy=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_axis_off()
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Add nodes and edges to start.", ha="center", va="center")
        return fig

    position = nx.spring_layout(graph, seed=7)
    scores = pagerank_scores or {node: 1 / max(1, graph.number_of_nodes()) for node in graph.nodes()}
    node_sizes = [900 + 4500 * scores.get(node, 0.0) for node in graph.nodes()]
    node_colors = ["#8ecae6" if graph.out_degree(node) else "#ffb703" for node in graph.nodes()]

    nx.draw_networkx_nodes(graph, position, node_size=node_sizes, node_color=node_colors, edgecolors="#264653", ax=ax)
    nx.draw_networkx_labels(graph, position, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(graph, position, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.6, alpha=0.75, ax=ax)

    if policy:
        policy_edges = [(node, target) for node, target in policy.items() if target]
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=policy_edges,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=24,
            width=3.2,
            edge_color="#d62828",
            ax=ax,
        )

    ax.set_title("Directed graph; larger nodes have higher PageRank")
    return fig


def plot_convergence(comparison):
    fig, ax = plt.subplots(figsize=(8, 4))
    for factor, run in comparison.items():
        ax.plot(range(1, len(run.history) + 1), run.history, marker="o", markersize=3, label=f"d={factor}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L1 error")
    ax.set_yscale("log")
    ax.set_title("PageRank convergence by damping factor")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def plot_walk_distribution(distribution):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = list(distribution.keys())
    values = list(distribution.values())
    ax.bar(names, values, color="#219ebc")
    ax.set_ylim(0, max(values + [0.1]) * 1.2)
    ax.set_ylabel("Visit share")
    ax.set_title("Random walk empirical distribution")
    return fig


def plot_rewards(rewards):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if rewards:
        window = min(20, len(rewards))
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(rewards, alpha=0.35, label="Episode reward")
        ax.plot(range(window - 1, window - 1 + len(smooth)), smooth, linewidth=2.5, label=f"{window}-episode average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Q-learning reward trend")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def show_matrix(matrix, nodes):
    rows = []
    for row_idx, source in enumerate(nodes):
        row = {"from": source}
        for col_idx, target in enumerate(nodes):
            row[target] = float(matrix[row_idx, col_idx])
        rows.append(row)
    st.dataframe(
        rows,
        column_config={node: st.column_config.NumberColumn(node, format="%.3f") for node in nodes},
        hide_index=False,
        use_container_width=True,
    )


st.set_page_config(page_title="PageRank and Random Walk Toolkit", layout="wide")
st.title("PageRank and Random Walk Lab")
st.caption("Build a directed graph, watch how movement behaves, and connect the same idea to ranking and learning.")

with st.sidebar:
    st.header("Graph")
    sample_name = st.selectbox("Start from a sample graph", list(SAMPLES))
    edge_text = st.text_area("Edges, one per line", value=SAMPLES[sample_name], height=230)
    extra_nodes = st.text_input("Extra isolated nodes, comma separated", value="")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

try:
    edges = parse_edges(edge_text)
    nodes = sorted({node for edge in edges for node in edge} | {node.strip() for node in extra_nodes.split(",") if node.strip()})
    graph = build_graph(nodes, edges)
except ValueError as error:
    st.error(str(error))
    st.stop()

nodes = sorted_nodes(graph)
if not nodes:
    st.warning("Add at least one node.")
    st.stop()

pagerank_default = pagerank_power_iteration(graph, damping=0.85)

left, right = st.columns([1.25, 1])
with left:
    st.pyplot(draw_graph(graph, pagerank_default.scores), clear_figure=True)

with right:
    st.subheader("Node Rankings")
    st.dataframe(
        [{"rank": rank, "node": node, "pagerank": score} for rank, (node, score) in enumerate(ranked_items(pagerank_default.scores), start=1)],
        hide_index=True,
        use_container_width=True,
    )
    st.write(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    st.write(f"Dangling nodes: {', '.join(dangling_nodes(graph)) or 'none'}")
    st.write(f"Absorbing states: {', '.join(absorbing_states(graph)) or 'none'}")

tabs = st.tabs(["Markov Chain", "Random Walk", "PageRank", "MDP and RL", "Report Flow"])

with tabs[0]:
    st.subheader("Transition Matrix")
    matrix = transition_matrix(graph, nodes)
    show_matrix(matrix, nodes)

    st.subheader("Communication Classes")
    class_rows = []
    for idx, item in enumerate(communication_classes(graph), start=1):
        class_rows.append({"class": idx, "nodes": ", ".join(item["nodes"]), "closed": item["closed"]})
    st.dataframe(class_rows, hide_index=True, use_container_width=True)

    st.subheader("Stationary Distribution")
    stationary, stationary_history = stationary_distribution(graph)
    st.dataframe(
        [{"node": node, "probability": value} for node, value in ranked_items(stationary)],
        hide_index=True,
        use_container_width=True,
    )
    st.write(f"Stationary iteration error after {len(stationary_history)} iterations: {stationary_history[-1]:.2e}" if stationary_history else "No iterations.")

with tabs[1]:
    st.subheader("Random Walk Simulation")
    walk_start = st.selectbox("Start node", nodes, key="walk_start")
    walk_steps = st.slider("Steps", 10, 2000, 250, step=10)
    walk = run_random_walk(graph, walk_start, walk_steps, seed=int(seed))
    st.pyplot(plot_walk_distribution(walk.visit_distribution), clear_figure=True)
    st.write("Path preview:")
    st.code(" -> ".join(walk.path[:80]) + (" -> ..." if len(walk.path) > 80 else ""))

with tabs[2]:
    st.subheader("PageRank and Damping Factor Comparison")
    comparison = compare_damping_factors(graph)
    st.pyplot(plot_convergence(comparison), clear_figure=True)

    rows = []
    for factor, run in comparison.items():
        for rank, (node, score) in enumerate(ranked_items(run.scores), start=1):
            rows.append({"damping": factor, "rank": rank, "node": node, "pagerank": score, "iterations": run.iterations})
    st.dataframe(rows, hide_index=True, use_container_width=True)

with tabs[3]:
    st.subheader("Decision Making Extension")
    st.write("Here the graph becomes a small travel problem. The red edges show the policy learned by Q-learning.")
    mdp_start = st.selectbox("Traveler start", nodes, key="mdp_start")
    mdp_goal = st.selectbox("Goal node", nodes, index=len(nodes) - 1, key="mdp_goal")
    policy = value_iteration_shortest_policy(graph, mdp_goal)
    rl = q_learning(graph, mdp_start, mdp_goal, episodes=300, seed=int(seed))

    left_rl, right_rl = st.columns([1, 1])
    with left_rl:
        st.pyplot(draw_graph(graph, pagerank_default.scores, policy=rl.policy), clear_figure=True)
    with right_rl:
        st.write("MDP policy from value iteration:")
        st.dataframe([{"state": node, "best action": action or "stop/dead end"} for node, action in policy.items()], hide_index=True, use_container_width=True)
        st.write("RL policy learned by Q-learning:")
        st.dataframe([{"state": node, "best action": action or "stop/dead end"} for node, action in rl.policy.items()], hide_index=True, use_container_width=True)
    st.pyplot(plot_rewards(rl.episode_rewards), clear_figure=True)

with tabs[4]:
    st.subheader("Submission Story")
    st.markdown(
        """
        1. **Markov Chain:** a directed graph becomes a transition matrix.
        2. **Random Walk:** repeated random movement estimates long-run behavior.
        3. **PageRank:** teleportation and damping make ranking robust even with dead ends.
        4. **MDP:** the traveler chooses actions instead of moving randomly.
        5. **RL:** Q-learning improves choices through trial and reward.
        """
    )
