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
        nearest_goal = objectives[0]
        min_distance = abs(start[0] - objectives[0][0]) + abs(start[1] - objectives[0][1])  # Initial Manhattan distance

        for i in range(1, len(objectives)):
            distance = abs(start[0] - objectives[i][0]) + abs(start[1] - objectives[i][1])  # Calculate Manhattan distance
            if distance < min_distance:
                min_distance = distance
                nearest_goal = objectives[i]

        # Use A* search to reach the nearest objective
        current_state = start
        frontier = []  # List of frontier nodes
        frontier.append((0, current_state))
        cost = {}
        cost[current_state] = 0
        parent_tree = {}

        while frontier:
            # Find the node with the lowest cost in the frontier
            min_index = 0
            min_cost = frontier[0][0]
            for i in range(1, len(frontier)):
                if frontier[i][0] < min_cost:
                    min_cost = frontier[i][0]
                    min_index = i
            current_state = frontier.pop(min_index)[1]

            if current_state == nearest_goal:
                break

            # Explore neighbors
            for neighbor in maze.getNeighbors(current_state[0], current_state[1]):
                new_cost = cost[current_state] + 1 + abs(neighbor[0] - nearest_goal[0]) + abs(neighbor[1] - nearest_goal[1])  # f=g-h, Manhattan
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    frontier.append((new_cost, neighbor))
                    parent_tree[neighbor] = current_state
        
        # Reconstruct the path for this objective and add it to the final path
        current = nearest_goal
        objective_path = [current]
        while current != start:
            current = parent_tree[current]
            objective_path.append(current)
        objective_path.reverse()
        path.extend(objective_path)  # Include the objective in the path

        # Mark the objective as visited and update the start position
        objectives_visited.append(nearest_goal)
        start = nearest_goal
        objectives.remove(nearest_goal)

    return path
