import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

st.set_page_config(page_title="AI Maze Solver", page_icon="🧩")

st.title("🧩 AI Maze Solver Using A* Search")
st.write("Find the shortest path from Start to Goal using Artificial Intelligence!")

ROWS = 15
COLS = 15
START = (0, 0)
GOAL = (14, 14)


def generate_maze():
    maze = np.zeros((ROWS, COLS), dtype=int)

    for r in range(ROWS):
        for c in range(COLS):
            if random.random() < 0.25:
                maze[r, c] = 1

    maze[0, :] = 0
    maze[:, COLS - 1] = 0

    return maze


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(maze):
    queue = [(0, START)]
    came_from = {}
    cost = {START: 0}
    explored = set()

    while queue:
        _, current = heapq.heappop(queue)

        if current in explored:
            continue

        explored.add(current)

        if current == GOAL:
            path = [GOAL]

            while path[-1] != START:
                path.append(came_from[path[-1]])

            path.reverse()
            return path, explored

        r, c = current

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if (
                0 <= nr < ROWS
                and 0 <= nc < COLS
                and maze[nr, nc] == 0
                and neighbor not in explored
            ):
                new_cost = cost[current] + 1

                if new_cost < cost.get(neighbor, float("inf")):
                    cost[neighbor] = new_cost
                    came_from[neighbor] = current

                    priority = new_cost + heuristic(neighbor, GOAL)
                    heapq.heappush(queue, (priority, neighbor))

    return None, explored


def display_maze(maze, path=None):
    display = maze.copy()

    if path:
        for r, c in path:
            display[r, c] = 2

    display[START] = 3
    display[GOAL] = 4

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        display,
        cmap=plt.matplotlib.colors.ListedColormap(
            ["white", "black", "green", "cyan", "red"]
        ),
        vmin=0,
        vmax=4
    )

    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    st.pyplot(fig)
    plt.close(fig)


if "maze" not in st.session_state:
    st.session_state.maze = generate_maze()

if st.button("Generate New Maze"):
    st.session_state.maze = generate_maze()

if st.button("Find Shortest Path"):
    path, explored = a_star(st.session_state.maze)

    if path:
        st.success("Goal reached successfully!")
        st.write("Shortest path length:", len(path) - 1, "steps")
        st.write("Cells explored:", len(explored))
        display_maze(st.session_state.maze, path)
    else:
        st.error("No path found. Generate a new maze.")

st.subheader("Maze")
st.write("🟦 Start | 🟥 Goal | 🟩 Shortest Path | ⬛ Obstacles")
display_maze(st.session_state.maze)
