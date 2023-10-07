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

    #print("objectives: " + str(objectives))
    #print("start: " + str(start))
    
    # Initialize FIFO data structure with the start state
    queue = [(start, [])]   # position, path
    
    while queue:
        current_pos, current_path = queue.pop(0)  # Pop from the front of the queue
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            
            # If all objectives are visited, return the path
            if not objectives:
                #print(current_path)
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

    # Initialize LIFO data structure with the start state
    stack = [(start, [])]  # position, path
    
    # Create a set to keep track of visited states
    visited = set()
    
    while stack:
        current_pos, current_path = stack.pop()  # Pop from the end of the stack
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            
            # If all objectives are visited, return the path
            if not objectives:
                return current_path
        
        # Mark the current state as visited
        visited.add(current_pos)
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            
            # Avoid revisiting visited positions
            if neighbor not in visited and maze.isValidMove(*neighbor):
                stack.append((neighbor, new_path))
    
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
    
    # Initialize a list for UCS, where each element is a tuple (cost, position, path)
    ucs_list = [(0, start, [])]
    
    # Create a set to keep track of visited states
    visited = set()
    
    while ucs_list:
        min_index = 0
        min_cost = ucs_list[0][0]

        # Sort the ucs_list data structure to find lowest cost
        for i in range(len(ucs_list)):
            cost = ucs_list[i][0]
            if cost < min_cost:
                min_cost = cost
                min_index = i
        min_state = ucs_list.pop(min_index)
        cost, current_pos, current_path = min_state  # Pop the state with the lowest cost
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            
            # If all objectives are visited, return the path
            if not objectives:
                return current_path
        
        # Mark the current state as visited
        visited.add(current_pos)
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            new_cost = cost + 1  # Assuming all step costs are equal to one
            
            # Avoid revisiting visited positions and update cost
            if neighbor not in visited and maze.isValidMove(*neighbor):
                ucs_list.append((new_cost, neighbor, new_path))
    
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
    
    # Initialize a list for A*, where each element is a tuple (fn, position, path)
    fn_start = abs(start[0] - objectives[0][0]) + abs(start[1] - objectives[0][1])  # f-cost = g-cost + h-cost
    astar_list = [(fn_start, start, [])]
    
    # Create a set to keep track of visited states
    visited = set()
    
    while astar_list:
        min_index = 0
        min_fn = astar_list[0][0]

        # Sort the astar_list data structure to find lowest cost
        for i in range(len(astar_list)):
            fn = astar_list[i][0]
            if fn < min_fn:
                min_fn = fn
                min_index = i
        min_state = astar_list.pop(min_index)
        fn, current_pos, current_path = min_state  # Pop the state with the lowest f cost
        
        # Check if the current position is an unvisited objective
        if current_pos in objectives:
            objectives.remove(current_pos)
            
            # If all objectives are visited, return the path
            if not objectives:
                return current_path
        
        # Mark the current state as visited
        visited.add(current_pos)
        
        # Get valid neighbors of the current position
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])
        
        for neighbor in neighbors:
            new_path = current_path + [current_pos]
            gn = len(new_path)  # g-cost is the number of steps taken
            
            # Calculate h-cost (hn) using the Manhattan distance directly
            hn = abs(neighbor[0] - objectives[0][0]) + abs(neighbor[1] - objectives[0][1])
            
            # Calculate f-cost (fn)
            fn = gn + hn
            
            # Avoid revisiting visited positions and update cost
            if neighbor not in visited and maze.isValidMove(*neighbor):
                astar_list.append((fn, neighbor, new_path))
    
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
