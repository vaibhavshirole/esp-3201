# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

import time

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "ucs": ucs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
    }.get(searchMethod)(maze)


class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position



def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    
    # Initialize FIFO data structure with the start state
    queue = [(start, [])]   # position, path
    while queue:
        
        # Pull next
        current_pos, current_path = queue.pop(0)
        
        # Check if univisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            if len(objectives) == 0:
                return current_path
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            
            # Don't allow to visit already visited
            if neighbor not in current_path:
                queue.append((neighbor, new_path))
    
    # Return empty list if no path found
    return []


def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()

    # Initialize LIFO data structure with the start state
    stack = [(start, [])]  # position, path
    while stack:

        # Pull next
        current_pos, current_path = stack.pop()
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            if len(objectives) == 0:
                return current_path
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            
            # Don't allow to visit already visited
            if neighbor not in current_path:
                stack.append((neighbor, new_path))
    
    # Return empty list if no path found
    return []


def ucs(maze):
    """
    Runs ucs for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    
    # Initialize UCS data structure
    ucs_list = [(0, start, [])]     # cost, position, path
    while ucs_list:
        
        # Inits
        min_index = 0
        min_cost = ucs_list[0][0]

        # Sort ucs_list data structure to find lowest cost
        for i in range(len(ucs_list)):
            cost = ucs_list[i][0]
            if cost < min_cost:
                min_cost = cost
                min_index = i
        min_state = ucs_list.pop(min_index)  # Pull lowest cost state
        cost, current_pos, current_path = min_state
        
        # Check if unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            if len(objectives) == 0:
                return current_path

        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])        
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            
            # Don't allow to visit already visited
            if neighbor not in current_path:
                ucs_list.append((cost+1, neighbor, new_path))  # Update cost
    
    # Return empty list if no path found
    return []


def manhattan_distance(point1, point2):
    """
    Calculates the Manhattan distance between two points.

    @param point1: First point: (x1, y1).
    @param point2: Second point: (x2, y2).

    @return distance: The Manhattan distance between the two points.
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    
    # Initialize A* data structure
    fn_start = manhattan_distance(start, objectives[0])  # f-cost = g-cost + h-cost
    astar_list = [(fn_start, start, [])]  # (fn, position, path)
    while astar_list:
        
        # Inits
        min_index = 0
        min_fn = astar_list[0][0]

        # Sort the astar_list data structure to find lowest cost
        for i in range(len(astar_list)):
            fn = astar_list[i][0]
            if fn < min_fn:
                min_fn = fn
                min_index = i
        min_state = astar_list.pop(min_index)
        fn, current_pos, current_path = min_state  # Pull lowest cost state
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            if len(objectives) == 0:
                return current_path

        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            
            gn = len(new_path)  # g-cost is the number of steps taken
            hn = manhattan_distance(neighbor, objectives[0]) # h-cost is heuristic cost
            fn = gn + hn    # f-cost is the cost
            
            # Don't allow to visit already visited
            if neighbor not in current_path:
                astar_list.append((fn, neighbor, new_path))
    
    # Return empty list if no path found
    return []


def astar_corner(maze):
    """
    Runs A* for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()

    # Define the heuristic (manhattan distance, but to each corner)
    def heuristic(state):
        # Calculate maximum Manhattan distance to all unvisited corners
        max_distance = 0
        for obj in objectives:
            found = False
            for i in range(len(objectives)):
                if state[0] == obj and i not in state[1]:
                    distance = abs(state[0][0] - obj[0]) + abs(state[0][1] - obj[1])
                    max_distance = max(max_distance, distance)
                    found = True
                    break
            if found:
                break

        return max_distance

    # Initialize A* data structure
    hn_start = heuristic((start, set(), []))
    astar_list = [(hn_start, 0, 0, start, set(), [])]  # (hn, gn, fn, position, collected, path)
    explored = set()

    while astar_list:

        # Inits
        min_index = 0
        hn_start = astar_list[0][0]

        # Sort the astar_list data structure to find the lowest cost
        for i in range(len(astar_list)):
            hn = astar_list[i][0]
            if hn < hn_start:
                hn_start = hn
                min_index = i
        min_state = astar_list.pop(min_index)
        _, gn, hn, current_pos, collected, path = min_state  # Pull lowest cost state

        # Check if all corners have been visited
        if len(collected) == len(objectives):
            return path

        # Check if current state has been explored
        if (current_pos, tuple(sorted(collected))) in explored:
            continue
        else:
            explored.add((current_pos, tuple(sorted(collected))))  # Mark as explored

        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        for neighbor in neighbors:
            new_path = path + [current_pos]  # Update path with the collected corners
            new_collected = set(collected)

            # Check if the new position collects a corner
            for i in range(len(objectives)):
                if neighbor == objectives[i] and i not in new_collected:
                    new_collected.add(i)
                    break

            gn = len(new_path)
            hn = heuristic((neighbor, new_collected, []))
            fn = gn + hn
            
            astar_list.append((hn, gn, fn, neighbor, new_collected, new_path))

    # If no path is found, return an empty list
    return []


#   CHANGED STRATEGY BECAUSE OLD METHOD WOULD SEARCH EVERYTHING AND TAKE TOO LONG!
def astar_multi(maze):
    """
    Runs A* search for finding the shortest path while visiting all objectives.

    @param maze: The maze to execute the search on.

    @return path: A list of tuples containing the coordinates of each state in the computed path.
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    objectives_visited = []  # To keep track of visited objectives
    path = []  # To store the final path

    while objectives:
        nearest_goal = objectives[0]  # Pre-determine the starting objective for efficiency
        min_distance = manhattan_distance(start, objectives[0])  # Initial Manhattan distance

        for i in range(1, len(objectives)):
            distance = manhattan_distance(start, objectives[i])  # Calculate Manhattan distance
            if distance < min_distance:
                min_distance = distance
                nearest_goal = objectives[i]

        current_state = start
        astar_list = []  # List of A* nodes
        fn = 0  # f-cost
        gn = 0  # g-cost
        hn = manhattan_distance(current_state, nearest_goal)  # h-cost (heuristic)

        astar_list.append((fn, gn, current_state))
        cost = {}
        cost[current_state] = 0
        parent_tree = {}

        while astar_list:
            
            # Inits
            min_index = 0
            min_fn = astar_list[0][0]

            # Sort the astar_list data structure to find the lowest cost
            for i in range(1, len(astar_list)):
                if astar_list[i][0] < min_fn:
                    min_fn = astar_list[i][0]
                    min_index = i
            current_state = astar_list.pop(min_index)[2]
            if current_state == nearest_goal:
                break

            # Explore neighbors
            for neighbor in maze.getNeighbors(current_state[0], current_state[1]):
                
                gn = cost[current_state] + 1
                hn = manhattan_distance(neighbor, nearest_goal)  # Calculate h-cost (heuristic)
                fn = gn + hn  # Calculate f-cost

                if neighbor not in cost or gn < cost[neighbor]:
                    cost[neighbor] = gn
                    astar_list.append((fn, gn, neighbor))
                    parent_tree[neighbor] = current_state
        
        # Add path to this objective to the final path
        current = nearest_goal
        objective_path = [current]
        while current != start:
            current = parent_tree[current]
            objective_path.append(current)
        objective_path.reverse()  # path was backwards
        path.extend(objective_path)  # Include the objective in the path

        # Mark objective visited and update start position
        objectives_visited.append(nearest_goal)
        start = nearest_goal
        objectives.remove(nearest_goal)

    return path
