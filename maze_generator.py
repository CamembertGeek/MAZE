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
    grid[0, 1] = 0 # Entry

    grid[height-1, width-2] = 0 # Exit

    return grid

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
            else:
                line += " "
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

    backtraking_maze = backtracking_maze_generator(30, 31)

    ascii_backtraking_maze = display_maze_ascii(backtraking_maze)
    for line in ascii_backtraking_maze:
        print(line)

    display_matplotlib_maze(backtraking_maze)

    
    