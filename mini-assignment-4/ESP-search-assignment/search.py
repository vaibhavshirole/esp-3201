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

    print("objectives: " + str(objectives))
    print("start: " + str(start))
    
    # Initialize the queue with the start state
    queue = [(start, [])]  # Each element in the queue is a tuple (position, path)
    
    while queue:
        current_pos, current_path = queue.pop(0)  # Pop from the front of the queue
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            
            # If all objectives are visited, return the path
            if not objectives:
                print(current_path)
                return current_path
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            
            # Avoid revisiting visited positions
            if neighbor not in current_path:
                queue.append((neighbor, new_path))
    
    # If no path is found, return an empty list to indicate failure
    return []


def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()

    print("objectives: " + str(objectives))
    print("start: " + str(start))
    
    # Initialize the stack with the start state and the remaining objectives
    stack = [(start, [], set(objectives))]  # Each element in the stack is a tuple (position, path, remaining_objectives)
    visited = set()  # To keep track of visited states
    
    while stack:
        current_pos, current_path, remaining_objectives = stack.pop()  # Pop from the end of the stack
        
        if not remaining_objectives:
            return current_path  # Found a path that covers all objectives
        
        visited.add(current_pos)
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            if neighbor not in visited:
                new_path = current_path + [current_pos]
                new_remaining = set(remaining_objectives)
                
                if neighbor in new_remaining:
                    new_remaining.remove(neighbor)
                
                stack.append((neighbor, new_path, new_remaining))
    
    # If no path is found, return an empty list to indicate failure
    return []


def ucs(maze):
    """
    Runs ucs for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    
    # Initialize the queue with the start state and the remaining objectives
    queue = [(0, start, [])]  # Each element in the queue is a tuple (cost, position, path)
    visited = set()  # To keep track of visited states
    
    while queue:
        # Sort the queue by cost in ascending order
        queue.sort(key=lambda x: x[0])
        
        cost, current_pos, current_path = queue.pop(0)  # Pop the element with the lowest cost
        
        if current_pos in objectives:
            objectives.remove(current_pos)
            current_path.append(current_pos)
            
            if not objectives:
                return current_path  # Found a path that covers all objectives
        
        visited.add(current_pos)
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            if neighbor not in visited:
                new_path = current_path + [current_pos]
                new_cost = cost + 1  # Assuming equal step costs
                
                queue.append((new_cost, neighbor, new_path))
    
    # If no path is found, return an empty list to indicate failure
    return []



def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    
    def heuristic(position, remaining_objectives):
        # A heuristic function that estimates the cost to reach the nearest remaining objective
        min_dist = float('inf')
        for obj in remaining_objectives:
            dist = abs(position[0] - obj[0]) + abs(position[1] - obj[1])
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    # Initialize the open set with the start state and the remaining objectives
    open_set = [(heuristic(start, objectives), 0, start, [])]  # (f_cost, g_cost, position, path)
    visited = set()  # To keep track of visited states
    
    while open_set:
        # Sort the open set by f_cost in ascending order
        open_set.sort(key=lambda x: x[0])
        
        f_cost, g_cost, current_pos, current_path = open_set.pop(0)  # Pop the element with the lowest f_cost
        
        if current_pos in objectives:
            objectives.remove(current_pos)
            current_path.append(current_pos)
            
            if not objectives:
                return current_path  # Found a path that covers all objectives
        
        visited.add(current_pos)
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            if neighbor not in visited:
                new_g_cost = g_cost + 1  # Assuming equal step costs
                new_f_cost = new_g_cost + heuristic(neighbor, objectives)
                
                open_set.append((new_f_cost, new_g_cost, neighbor, current_path + [current_pos]))
    
    # If no path is found, return an empty list to indicate failure
    return []


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
