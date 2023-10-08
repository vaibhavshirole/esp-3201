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
    Runs A* for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    num_objectives = len(objectives)

    # Define the heuristic function
    def heuristic(state):
        # Calculate the maximum Manhattan distance to an unvisited corner
        max_distance = 0
        for obj in objectives:
            found = False
            for i in range(num_objectives):
                if state[0] == obj and i not in state[1]:
                    distance = abs(state[0][0] - obj[0]) + abs(state[0][1] - obj[1])
                    max_distance = max(max_distance, distance)
                    found = True
                    break
            if found:
                break
        return max_distance

    # Initialize the priority queue with the starting state
    fn_start = heuristic((start, set(), []))
    astar_list = [(fn_start, 0, start, set(), [])]

    # Initialize the explored set
    explored = set()

    while astar_list:
        min_index = 0
        fn_start = astar_list[0][0]

        # Find the state with the lowest priority
        for i in range(len(astar_list)):
            fn = astar_list[i][0]
            if fn < fn_start:
                fn_start = fn
                min_index = i

        # Pop the state with the lowest priority
        _, cost, current_pos, collected, path = astar_list.pop(min_index)

        # Check if all corners have been visited
        if len(collected) == num_objectives:
            #print(path)
            return path

        # Check if the current state has already been explored
        if (current_pos, tuple(sorted(collected))) in explored:
            continue

        # Mark the current state as explored
        explored.add((current_pos, tuple(sorted(collected))))

        # Generate successor states (valid neighbors)
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])

        for neighbor in neighbors:
            new_pos = neighbor
            new_collected = set(collected)

            # Check if the new position collects a corner
            for i in range(num_objectives):
                if new_pos == objectives[i] and i not in new_collected:
                    new_collected.add(i)
                    break

            # Calculate the new cost and update the list
            new_cost = cost + 1
            new_fn = new_cost + heuristic((new_pos, new_collected, []))
            new_path = path + [(new_pos)]  # Update path with the collected corners
            astar_list.append((new_fn, new_cost, new_pos, new_collected, new_path))

    # If no path is found, return an empty list
    return []


import heapq

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.getStart()
    objectives = maze.getObjectives()
    num_objectives = len(objectives)

    # Initialize a table to store MST lengths for sets of dots
    mst_table = {}

    def compute_mst_length(dot_set):
        """
        Compute the length of the Minimum Spanning Tree (MST) for a set of dots using Prim's algorithm.

        @param dot_set: A set of dot coordinates.

        @return mst_length: The length of the MST.
        """
        if len(dot_set) <= 1:
            return 0

        # Create a priority queue for Prim's algorithm
        pq = [(0, start)]
        mst_length = 0
        connected_dots = set()

        while pq:
            cost, current = heapq.heappop(pq)

            if current in connected_dots:
                continue

            connected_dots.add(current)
            mst_length += cost

            for dot in dot_set - connected_dots:
                heapq.heappush(pq, (abs(current[0] - dot[0]) + abs(current[1] - dot[1]), dot))

        return mst_length

    # Define the heuristic function
    def heuristic(state):
        # Check if there are any dots remaining
        if not state[1]:
            return 0  # No remaining dots, heuristic value is 0

        # Calculate the MST length for the remaining dots
        remaining_dots = set(objectives)  # Use a set for dot_set
        for dot_index in state[1]:
            remaining_dots.remove(objectives[dot_index])  # Remove dot using dot_index

        if tuple(remaining_dots) in mst_table:
            mst_length = mst_table[tuple(remaining_dots)]
        else:
            mst_length = compute_mst_length(remaining_dots)
            mst_table[tuple(remaining_dots)] = mst_length

        # Calculate the nearest dot's distance
        nearest_dot_distance = min(abs(state[0][0] - objectives[dot_index][0]) + abs(state[0][1] - objectives[dot_index][1]) for dot_index in state[1])

        return nearest_dot_distance + mst_length

    # Initialize the priority queue with the starting state
    fn_start = heuristic((start, set(), []))
    astar_list = [(fn_start, 0, start, set(), [])]  # Use a set for collected

    # Initialize the explored set
    explored = set()

    while astar_list:
        min_index = 0
        fn_start = astar_list[0][0]

        # Find the state with the lowest priority
        for i in range(len(astar_list)):
            fn = astar_list[i][0]
            if fn < fn_start:
                fn_start = fn
                min_index = i

        # Pop the state with the lowest priority
        _, cost, current_pos, collected, path = astar_list.pop(min_index)

        # Check if all dots have been visited
        if len(collected) == num_objectives:
            return path

        # Check if the current state has already been explored
        if (current_pos, tuple(sorted(collected))) in explored:
            continue

        # Mark the current state as explored
        explored.add((current_pos, tuple(sorted(collected))))

        # Generate successor states (valid neighbors)
        neighbors = maze.getNeighbors(current_pos[0], current_pos[1])

        for neighbor in neighbors:
            new_pos = neighbor
            new_collected = set(collected)  # Use a set for collected

            # Check if the new position collects a dot
            for i, dot in enumerate(objectives):
                if new_pos == dot and i not in new_collected:
                    new_collected.add(i)
                    break

            # Calculate the new cost and update the list
            new_cost = cost + 1
            new_fn = new_cost + heuristic((new_pos, new_collected, []))
            new_path = path + [(new_pos)]  # Update path with the collected dots
            astar_list.append((new_fn, new_cost, new_pos, new_collected, new_path))

    # If no path is found, return an empty list
    return []
