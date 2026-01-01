import torch

training_file_path = "/home/louis/Documents/Programs/365/MAZE/Training/npy_files/training_grid.pt"
training_solved_file_path = "/home/louis/Documents/Programs/365/MAZE/Training/npy_files/solved_grid.pt"

grids = torch.load(training_file_path, map_location="cpu") # grids is a dictionary

# print(grids.keys())
print(len(grids))
print(list(grids.keys())[:5])

first_grid_dict = grids['000001']
print(first_grid_dict.keys())

first_grid = first_grid_dict['grid']
first_grid_entry = first_grid_dict['entry']
first_grid_exit = first_grid_dict['exit']

print(first_grid_entry)
print(first_grid_exit)
print(first_grid.shape)
print(first_grid.dtype)
print(first_grid.min(), first_grid.max())
print(first_grid)
print()




solved_grids = torch.load(training_solved_file_path, map_location="cpu")

print(len(solved_grids))
print(list(solved_grids.keys())[:5])

first_solution = solved_grids['000001']
print(first_solution.keys())

first_grid_solved = first_solution['solution']
print(first_grid_solved.shape)
print(first_grid_solved.dtype)
print(first_grid_solved.min(), first_grid_solved.max())
print(first_grid_solved)