import numpy as np
import os
from os import listdir
from os.path import isfile, join
import torch

def npz_to_torch_tensor_train_input(input_path: str, output_path: str):
    """
    This function convert each of the npz files in the input directory in one torch tensor file.

    PARAMETERS:
    -----------
    input_path : str
        Path to the input directory with npz files.
    output_path : str
        Paht to the output directory where the npy file will be created.
    """
    all_tensor_dictionnary = {}

    for file_name in sorted(listdir(input_path)):

        file_path = join(f"{input_path}/", f"{file_name}")

        name, _ = os.path.splitext(file_name) # To isolate the file name.
        id = name.rsplit("_", 1)[-1] # To isolate the grid id.

        tensor_dictionnary = {}

        with np.load(file_path) as npz_file:
            
            tensor_dictionnary["grid"] = torch.from_numpy(npz_file["grid"])
            tensor_dictionnary["entry"] = torch.from_numpy(npz_file["entry"])
            tensor_dictionnary["exit"] = torch.from_numpy(npz_file["exit"])

            all_tensor_dictionnary[id] = tensor_dictionnary


    torch.save(all_tensor_dictionnary, output_path + "/training_grid.pt")

def npz_to_torch_tensor_train_output(input_path: str, output_path: str):
    """
    This function convert each of the npz files in the sol_grid directory in one torch tensor file.

    PARAMETERS:
    -----------
    input_path : str
        Path to the input directory with npz files.
    output_path : str
        Paht to the output directory where the npy file will be created.
    """
    all_tensor_dictionnary = {}

    for file_name in sorted(listdir(input_path)):

        file_path = join(f"{input_path}/", f"{file_name}")

        name, _ = os.path.splitext(file_name) # To isolate the file name.
        id = name.rsplit("_", 1)[-1] # To isolate the grid id.

        tensor_dictionnary = {}

        with np.load(file_path) as npz_file:
            
            tensor_dictionnary["solution"] = torch.from_numpy(npz_file["solution"])

            all_tensor_dictionnary[id] = tensor_dictionnary

    torch.save(all_tensor_dictionnary, output_path + "/solved_grid.pt")




if __name__ == "__main__":
    
    # Convertion of the initail grid from npz to torch tensor.

    # Input_path = "/home/louis/Documents/Programs/365/MAZE/Training/grid"
    # Output_path = "/home/louis/Documents/Programs/365/MAZE/Training/npy_files"

    # npz_to_torch_tensor_train_input(Input_path, Output_path)




    # Convertion of the solution of the grid from npz to torch tensor.

    # Input_path = "/home/louis/Documents/Programs/365/MAZE/Training/sol_grid"
    # Output_path = "/home/louis/Documents/Programs/365/MAZE/Training/npy_files"

    # npz_to_torch_tensor_train_output(Input_path, Output_path)