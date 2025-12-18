import numpy as np
import random
import matplotlib.pyplot as plt


def generate_maze(dim:int, p:float):
    """
    Docstring for generate_maze
    
    PARAMETERS:
    -----------
    dim : int
        The dimension of the maze.
    p : float
        The probability of having a wall.

    RETURN:
    ------
    grid : 
        The maze grid.
    """

    init_grid = np.zeros(shape=(dim, dim))

    for i in range(dim):
        for j in range(dim):

            x = random.random()

            if x <= p:
                init_grid[(i,j)] = 1 # wall
            else:
                init_grid[(i,j)] = 0 # path

    return init_grid

def backtracking_maze_generator(height: int, width: int):
    """
    This function generate a maze grid using backtracking.
    
    PARAMETERS:
    ----------
    height : int
        The height of the grid.
    width : int
        The width of the grid.
    
    RETURN:
    ------
    grid : np.array
        The maze grid.
    """

    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1

    grid = np.ones(shape=(height, width), dtype=int)

    starting_point = (1,1)

    grid[starting_point] = 0 # Initialization of the starting point as path

    stack = [starting_point] # Stocking of all the visited point, it's a tuple list

    while stack:

        neighbors = [] # Initialization of the neighbors list

        current = stack[-1] # Current cell

        ci, cj = current 

        # We test all direction to see if there is neighbors
        # UP
        UPi, UPj = ci - 2, cj

        if 1 <= UPi < height-1 and 1 <= UPj < width-1:
            if grid[(UPi, UPj)] == 1:
                neighbors.append((UPi, UPj))
            
        # DOWN
        Di, Dj = ci + 2, cj

        if 1 <= Di < height-1 and 1 <= Dj < width-1:
            if grid[(Di, Dj)] == 1:
                neighbors.append((Di, Dj))

        # Left
        Li, Lj = ci, cj - 2

        if 1 <= Li < height-1 and 1 <= Lj < width-1:
            if grid[(Li, Lj)] == 1:
                neighbors.append((Li, Lj))

        # RIGHT
        Ri, Rj = ci, cj + 2

        if 1 <= Ri < height-1 and 1 <= Rj < width-1:
            if grid[(Ri, Rj)] == 1:
                neighbors.append((Ri, Rj))

        # We select a random neighbor in the list and we "dig" to it if the neighbor list is not empty
        if neighbors:
            nb_neighbor = len(neighbors)
            x = random.randint(0, nb_neighbor - 1) # We select a random neighbor

            neighbor = neighbors[x]
            ni, nj = neighbor

            grid[(ni, nj)] = 0 # We dig the neighbor cell

            mi, mj = (ni + ci) // 2, (nj + cj) // 2
            grid[(mi, mj)] = 0 # we dig the intermediate cell

            stack.append(neighbor) # We stack the neigbor cell to start for here in the new cycle

        else:
            stack.pop()

    # Creat an entry and an exit
    # First we stock the possible entry and exit cells on each wall

    doors_top = []
    doors_bottom = []
    doors_left = []
    doors_right = []

    # Top wall
    for j in range(1, width - 1):
        if grid[1, j] == 0:
            doors_top.append((0, j))

    # Bottom wall
    for j in range(1, width - 1):
        if grid[height - 2, j] == 0:
            doors_bottom.append((height - 1, j))

    # Left wall
    for i in range(1, height - 1):
        if grid[i, 1] == 0:
            doors_left.append((i, 0))

    # Right wall
    for i in range(1, height - 1):
        if grid[i, width - 2] == 0:
            doors_right.append((i, width - 1))


    # Select a random entry wall
    walls = ["top", "bottom", "left", "right"]
    entry_wall = random.choice(walls)

    # Select the entry cell
    if entry_wall == "top":
        entry_cell = random.choice(doors_top)

    elif entry_wall == "bottom":
        entry_cell = random.choice(doors_bottom)

    elif entry_wall == "left":
        entry_cell = random.choice(doors_left)

    elif entry_wall == "right":
        entry_cell = random.choice(doors_right)

    # Open the entry
    grid[entry_cell] = 0


    # Select the possible exit cells according to the entry wall
    exit_candidates = []

    # If entry is on the top wall
    if entry_wall == "top":
        # The exit can be anywhere on the bottom wall
        exit_candidates.extend(doors_bottom)

        # Or on the second half of the left and right walls
        for cell in doors_left:
            if cell[0] >= height // 2:
                exit_candidates.append(cell)

        for cell in doors_right:
            if cell[0] >= height // 2:
                exit_candidates.append(cell)

    # If entry is on the bottom wall
    elif entry_wall == "bottom":
        exit_candidates.extend(doors_top)

        for cell in doors_left:
            if cell[0] <= height // 2:
                exit_candidates.append(cell)

        for cell in doors_right:
            if cell[0] <= height // 2:
                exit_candidates.append(cell)

    # If entry is on the left wall
    elif entry_wall == "left":
        exit_candidates.extend(doors_right)

        for cell in doors_top:
            if cell[1] >= width // 2:
                exit_candidates.append(cell)

        for cell in doors_bottom:
            if cell[1] >= width // 2:
                exit_candidates.append(cell)

    # If entry is on the right wall
    elif entry_wall == "right":
        exit_candidates.extend(doors_left)

        for cell in doors_top:
            if cell[1] <= width // 2:
                exit_candidates.append(cell)

        for cell in doors_bottom:
            if cell[1] <= width // 2:
                exit_candidates.append(cell)


    # Remove the entry cell from exit candidates if needed
    if entry_cell in exit_candidates:
        exit_candidates.remove(entry_cell)

    # Select the exit cell
    exit_cell = random.choice(exit_candidates)

    # Open the exit
    grid[exit_cell] = 0

    # grid[0, 1] = 0 # Entry

    # grid[height-1, width-2] = 0 # Exit

    return grid

def add_loops(grid: np.array, loop_factor: float = 0.1):
    """
    This function take the "perfect" maze generated by the function backtracking_maze_generator and delet some walls to creat loops inside of the maze.

    PARAMETERS:
    ----------
    grid : np.array
        The "perfect" maze grid.
    loop_factor : float
        A factor comprise between 0 and 1 to creat a certain amount of loops, default set to 10%.

    RETURN:
    ------
    loop_grid : np.array
        The new maze grid with loops added.
    """
    if not 0 <= loop_factor <= 1:
        raise ValueError("loop_factor must be between 0 and 1.")


    loop_grid = grid.copy()

    (height, width) = grid.shape

    # Create a list of walls excluing the borders.
    walls = []
    
    for i in range(1, height-1):
        for j in range(1, width-1):

            if grid[(i, j)] == 1:
                if (grid[i, j-1] == 0 and grid[i, j+1] == 0) or (grid[i-1, j] == 0 and grid[i+1, j] == 0):
                    walls.append((i, j))

    # Select how many walls will be delleted.
    k = int(loop_factor * len(walls)) 

    # Select random walls throuht the wall list and delet them.
    random.shuffle(walls)
    walls_to_delet = walls[:k]

    # Delet the walls selected.
    for wall in walls_to_delet:
        loop_grid[wall] = 0

    return loop_grid

def maze_solver(grid: np.array):
    """
    This function solve the maze and save the path.

    PARAMETER:
    ---------
    grid : np.array
        The maze grid.
    
    RETURN:
    ------
    solved_grid : np.array
        The solved maze grid.
    """
    (height, width) = grid.shape

    # First find the entry and exit of the maze on the border.
    entry_exit = []

    for i in range(width):
        if grid[(0, i)] == 0: # Verification of the top border
            entry_exit.append((0, i))

        if grid[(-1, i)] == 0: # Verification of the bottom border
            entry_exit.append((height-1, i))
    
    for i in range(1, height-1):
        if grid[(i, 0)] == 0: # Verification of the left border
            entry_exit.append((i, 0))
        
        if grid[(i, -1)] == 0: # Verification of the right border
            entry_exit.append((i, width-1))

     
    # Find the path between the entry and the exit. For that we use a BFS methode.
    entry_point = entry_exit[0] # We select our entry.
    exit_point = entry_exit[1]

    visited = set() # List of visited cell, start with the entry point.
    visited.add(entry_point)

    parent = {} # Dictionary use later to reconstruct the path.
    parent[entry_point] = None

    queue = [entry_point] # List of the cell to visit

    found = False 

    while queue and found is False:

        current = queue.pop(0)
        ni, nj = current

        if current == exit_point:
            found = True # The exit is found
            break

        for direction in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            mi, mj = direction

            ci = ni + mi
            cj = nj + mj

            neighbor = (ci, cj)

            if 0 <= ci < height and 0 <= cj < width:
                if grid[neighbor] == 0:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)

    if not found:
        print(f"There is no solution to the maze.")


    # Reconstruction of the path
    path = []
    node = exit_point

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()


    # write the path on the grid
    solved_grid = grid.copy()
    for solved_cell in path:
        solved_grid[solved_cell] = 2

    return solved_grid


def display_maze_ascii(grid:np.array):
    """
    Docstring for display_maze_ascii
    
    PARAMETER:
    ---------
    grid : np.array
        THe maze grig.
    
    RETURN:
    ------
    ascii_grid : np.array
        The ASCII maze grid.
    """
    ascii_grid = []

    for row in grid:
        line=""
        for cell in row:
            if cell == 1:
                line += "█"
            elif cell == 0:
                line += " "
            elif cell == 2:
                line += "£"
        ascii_grid.append(line)

    return ascii_grid

def display_matplotlib_maze(grid:np.array):
    """
    Docstring for display_matplotlib_maze
    
    PARAMETER:
    ---------
    grid : np.array
        THe maze grig.
    """
    plt.figure()
    plt.imshow(grid)
    plt.show()

if __name__ == "__main__":

    # maze = generate_maze(10, 0.7)

    # print(maze)

    # ascii_maze = display_maze_ascii(maze)

    # for line in ascii_maze:
    #     print(line)

    # display_matplotlib_maze(maze)

    backtraking_maze = backtracking_maze_generator(50, 51)

    # ascii_backtraking_maze = display_maze_ascii(backtraking_maze)
    # for line in ascii_backtraking_maze:
    #     print(line)

    # display_matplotlib_maze(backtraking_maze)



    maze_with_loops = add_loops(backtraking_maze)

    ascii_maze_with_loops = display_maze_ascii(maze_with_loops)
    for line in ascii_maze_with_loops:
         print(line)

    display_matplotlib_maze(maze_with_loops)




    print(f"Solution")






    solved_maze = maze_solver(maze_with_loops)

    ascii_solved_maze = display_maze_ascii(solved_maze)
    for line in ascii_solved_maze:
         print(line)

    display_matplotlib_maze(solved_maze)

    
    