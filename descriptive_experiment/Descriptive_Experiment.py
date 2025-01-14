import os
import shutil
import sys

import pandas as pd
import GPUtil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Parameters(object):
    def __init__(self, path_config):
        """
        Initialize the Parameters class.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        # Load JSON configuration file
        if not os.path.isfile(path_config):
            raise FileNotFoundError(f"The JSON configuration file does not exist: {path_config}")
        try:
            with open(path_config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading JSON configuration file file {path_config}: {e}")

        # Set parameters
        self.seed = config["seed"]
        self.model_address = config["model_address"]
        self.input_text = config["input_text"]
        self.path_output = config["path_output"]
        self.max_length = config["max_length"]
        self.temperature = config["temperature"]  #temperature=0.7

        # Other configurations
        self.step = config["step"]
        self.pad_list = config["pad_list"]
        self.experiment_name = config["experiment_name"]
        self.time = config["time"]
        
        # Verify paths and raise errors if necessary
        self._verify_paths()

    def _verify_paths(self):
        """
        Verify the existence of required directories and files. Raise errors if any validation fails.
        """
        # Check if the model address exists
        if not os.path.isdir(self.model_address):
            raise FileNotFoundError(f"The model address does not exist: {self.model_address}")

        # Check if the output directory exists or can be created
        if not os.path.exists(self.path_output):
            try:
                os.makedirs(self.path_output)
            except Exception as e:
                raise RuntimeError(f"Failed to create the output directory: {self.path_output}\nError: {e}")

class Data(object):
    def __init__(self, ps):
        """
        Initialize the Data class.

        Args:
            ps: Parameters object containing paths and configuration settings.
        """
        # File paths
        # Standard file: Records output from the standard model with no mutation
        self.path_std_file = os.path.join(ps.path_output, f'/{ps.experiment_name}_std_{ps.time}.txt')

        # Code file: Records the unique integer code assigned to each unique output (phenotype) from mutations
        self.path_code_file = os.path.join(ps.path_output, f'/{ps.experiment_name}_{{}}_code_{ps.time}.txt')  # {{mid}}

        # Log file: Records each non-silent mutation in the format {pad} {mid}:{loc1},{loc2}:{code of phenotype}
        self.path_log_file = os.path.join(ps.path_output, f'/{ps.experiment_name}_{{}}_log_{ps.time}.txt')  # {{mid}}

        # Finish file: Records the site (loc1, loc2) of the last mutation calculated; (loc1,loc2) mutation location on matrix
        self.path_finished_file = os.path.join(ps.path_output, f'/{ps.experiment_name}_{{}}_{ps.time}.txt')  # {{mid}}
        
        # Done file: Records mid if the calculation for the mid matrix is complete
        self.path_done_file = os.path.join(ps.path_output, f'/{ps.experiment_name}_done_{ps.time}.txt')

        # Mutation information record
        self.pad_dict = {}
        self.var2k_dict = {}
        return None

def check_done(dt, mid):
    """
    Check if the given mid (matrix ID) has already been processed.

    Args:
        dt: Data object containing file paths.
        mid: Matrix id.

    Returns:
        bool: True if processing is complete, False otherwise.
    """
    if os.path.isfile(dt.path_done_file):
        with open(dt.path_done_file, 'r') as raw_file:
            done_list = raw_file.readline()
        if mid in done_list:
            return True
    return False

def model_load_func(ps, dt):
    """
    Load the model and tokenizer into memory.

    Args:
        ps: Parameters object containing model settings.
        dt: Data object containing GPU allocation and processing settings.
    """
    # Initialize tokenizer
    ps.tokenizer = AutoTokenizer.from_pretrained(ps.model_address, padding_side="left")
    ps.tokenizer.pad_token_id = 0 if ps.tokenizer.pad_token_id is None else ps.tokenizer.pad_token_id
    ps.tokenizer.bos_token_id = 1  # Ensure BOS token ID is set
    dt.gpu_id = _get_gpu()
    
    # Load the model onto the GPU
    ps.model=AutoModelForCausalLM.from_pretrained(ps.model_address).to(f"cuda:{dt.gpu_id}")
    ps.model.generation_config.pad_token_id = ps.tokenizer.pad_token_id

    # Record token of input
    ps.inputs = ps.tokenizer.encode(ps.input_text, return_tensors="pt")
    ps.inputs = ps.inputs.to(f"cuda:{dt.gpu_id}")
    ps.para_dict = ps.model.state_dict()
    return None

def standard_output_func(ps, dt):
    """
    Collect the answers from the standard model (non-mutated state).

    Args:
        ps: Parameters object containing model configurations.
        dt: Data object containing input and output paths.
    """
    if os.path.isfile(dt.path_std_file):
        with open(dt.path_std_file, 'r') as std_file:
            dt.std = std_file.read()
    else:
        torch.manual_seed(ps.seed)  # Set random seed for reproducibility
        output_token  = ps.model.generate(ps.inputs, max_length=ps.max_length, num_return_sequences=1, do_sample=True, temperature=ps.temperature)[0]
        dt.std_output_token_str =  ' '.join([str(item) for item in output_token.tolist()])
        output = ps.tokenizer.decode(output_token)
        output = ''.join([i if ord(i)<128 else 'ord({})'.format(ord(i)) for i in output])  # remove ascii
        dt.std = output
        with open(dt.path_std_file, 'w') as std_file:
            std_file.write(dt.std)    
    return None

def file_func(ps, dt, mid):
    """
    Make a code dictionary of processed mutations on mid matrix.
    Extract the last processed mutation location.

    Args:
        ps: Parameters object containing configuration settings.
        dt: Data object containing file paths and processing states.
        mid: Matrix id.

    Returns:
        tuple: The location (loc1, loc2) of mutation to start.
    """
    # Log file
    _log_file_func(ps, dt, mid)
    # Record mutation codes
    _var2k_file_func(ps, dt, mid)
    # Retrieve the site of the last calculated mutation
    loc1, loc2 = _stop_site_func(ps, dt, mid)
    return loc1, loc2

def one_mid_func(ps, dt, mid, w_st, h_st):
    """
    Process the specified intermediate layer parameters, performing padding, computation, and restoration.

    Args:
        ps: Parameters object containing the parameter dictionary and configuration.
        dt: Data object containing the padding dictionary and path information.
        mid: Matrix id.
        loc1 (int): Starting loc1 position.
        loc2 (int): Starting loc2 position.
    """
    print(f"{mid} start:")
    device = ps.para_dict[mid].get_device()
    ps.inputs.to(device)
    loc1_size, loc2_size = ps.para_dict[mid].size()
    
    # Initialize padding data
    dt.pad_dict = _initialize_pad_dict(ps, device, mid)

    # Process the parameter matrix all sites with one loc2
    for loc2_idx in range(loc2, loc2_size, ps.step):
        _process_one_loc2(ps, dt, mid, loc1_size, loc2_size, loc2_idx, loc1)
        loc1 = 0  # Reset loc1 to the beginning after completing a loc2

    # Record the matrix id when done
    with open(dt.path_done_file, 'a') as output_file:
        output_file.write(f'{mid}\n')

    print(f"{mid} done:")
    return None

def _initialize_pad_dict(ps, device, mid):
    """
    Initialize the padding dictionary, including zero-padding, minimum-value padding, and maximum-value padding.

    Args:
        ps: Parameters object containing the parameter dictionary and configuration.
        device: The current device (CPU or GPU).
        mid: Matrix id.

    Returns:
        dict: A dictionary containing padding matrices for zero, minimum, and maximum values.
    """
    step = ps.step
    pad_dict = {}
    param_matrix = ps.para_dict[mid]

    pad_dict['zero'] = torch.zeros(step, step).to(device)  # Zero padding
    pad_dict['max'] = torch.full((step, step), torch.max(param_matrix), device=device)  # Max-value padding
    pad_dict['min'] = torch.full((step, step), torch.min(param_matrix), device=device)  # Min-value padding

    return pad_dict

        
def _process_one_loc2(ps, dt, mid, loc1_size, loc2_size, loc2_idx, loc1):
    """
    Process all sites with the same loc2 on the parameter matrix, performing padding and restoration operations block by block.

    Args:
        ps: Parameters object containing the parameter dictionary and configuration.
        dt: Data object containing the padding dictionary and path information.
        mid: Matrix id.
        loc1_size (int): size of loc1 of the matrix.
        loc2_size (int):size of loc2 of the matrix.
        loc2_idx (int): Current loc2 index.
        loc1 (int): Starting loc1 site.
    """
    step = ps.step
    param_matrix = ps.para_dict[mid]

    for loc1_idx in range(loc1, loc1_size, step):
        # Clone the original block data
        org_p = torch.clone(param_matrix[loc1_idx:loc1_idx + step, loc2_idx:loc2_idx + step])

        # Perform operations for each padding type
        for pad_type, pad_matrix in dt.pad_dict.items():
            _output_func(ps, dt, pad_type, mid, loc1_idx, loc2_idx)

        # Restore the original data
        param_matrix[loc1_idx:loc1_idx + step, loc2_idx:loc2_idx + step] = org_p

        # Mark the current block as processed
        _finished_func(ps, dt, loc1_idx, loc2_idx)
    return None


def _get_gpu():
    GPUs = GPUtil.getGPUs()
    # return id of GPU with largest mem
    return sorted(GPUs, key=lambda x: x.memoryFree, reverse=True)[0].id

def _log_file_func(ps, dt, mid):
    """
    Append a marker on log filefor tracking the last processed location

    Args:
        ps: Parameters object containing configuration settings.
        dt: Data object containing file paths and processing states.
        mid: Matrix id.
    """
    dt.path_log_file = dt.path_log_file.format(mid)
    if os.path.isfile(dt.path_log_file):
        with open(dt.path_log_file, 'a') as output_file:
            # Add a marker to indicate the point of breakdown from the last session
            output_file.write("%%%%%%\n") 
    return None

def _var2k_file_func(ps, dt, mid):
    """
    Retrieve mutation codes from code file

    Args:
        ps: Parameters object containing configuration settings.
        dt: Data object containing file paths and processing states.
        mid: Matrix id.
    """
    dt.path_code_file = dt.path_code_file.format(mid)
    if not os.path.isfile(dt.path_code_file):
        # If the code file does not exist, initialize it
        with open(dt.path_code_file, 'w') as output_file:
            standard_output_func(ps, dt)
            output=("seed:{} step: {}\nstandard output:\n{}\n%%%%%%\n".format(ps.seed, ps.step, dt.std))
            output_file.write(output)
    else:
        # Extract existing codes
        with open(dt.path_code_file, 'r') as raw_file:
            temp_list = raw_file.read().split('\n%%%%%%\n')
            std = temp_list[0]
            std = '\n'.join(std.split('\n')[2:])  #remove the first 2 lines
            temp_list.pop(0)
            temp_list.pop(-1)
            if temp_list:
                _var2k_dict_func(ps, dt, temp_list)
            if std != dt.std:
                x = input("standard input different! Need to redo the whole set!")
    return None


def _var2k_dict_func(ps, dt, temp_list):
    """
    Extract codes recorded in the code file and populate the variable dictionary.

    Args:
        ps: Parameters object.
        dt: Data object containing the variable2code dictionary.
        temp_list: List of data from the code file.
    """
    for item in temp_list:
        var = '\n'.join(item.split('\n')[1:])
        key = item.split('\n')[0].split(':')[1]
        dt.var2k_dict[var] = key
    return None

def _stop_site_func(ps, dt, mid):
    """
    Retrieve the site of the last calculated mutation on matrix

    Args:
        ps: Parameters object containing configuration settings.
        dt: Data object containing file paths and processing states.
        mid: Matrix id.
    """
    loc1, loc2 = [0, 0]
    dt.path_finished_file = dt.path_finished_file.format(mid)
    if os.path.isfile(dt.path_finished_file):
        with open(dt.path_finished_file, 'r') as raw_file:
            loc1, loc2 = [int(item) for item in raw_file.readline().split(',')]
            loc1 += ps.step  # Advance to the next step
    return loc1, loc2

def _finished_func(ps, dt, w, h):
    """
    Records the site of the most recently processed mutation.

    Args:
        ps: Parameters object.
        dt: Data object containing file paths.
        loc1 (int): Starting loc1 site.
        loc2 (int): Starting loc2 site.
    """
    with open(dt.path_finished_file, 'w') as output_file:
        output_file.write("{},{}".format(w, h))
    return None

def _output_func(ps, dt, pad, mid, loc1, loc2):
    """
    Handles mutation processing and logging results.

    Args:
        ps: Parameters object containing model and parameter configurations.
        dt: Data object containing padding and file paths.
        pad: The type of padding applied (e.g., zero, min, max).
        mid: Matrix id.
        loc1 (int): Starting loc1 site.
        loc2 (int): Starting loc2 site.
    """
    ps.para_dict[mid][loc1:loc1+ps.step, loc2:loc2+ps.step] = dt.pad_dict[pad][:,:]
    # Set random seed for reproducibility
    torch.manual_seed(ps.seed)
    
    # Generate output tokens
    output_token = ps.model.generate(ps.inputs, max_length=ps.max_length, num_return_sequences=1, do_sample=True, temperature=0.7)[0]
    key = _key_func(ps, dt, mid, output_token)
    if key != 'std':
        with open(dt.path_log_file, 'a') as log_file:
            log_file.write("{} {}:{},{}:{}\n".format(pad, mid, w, h, key))
    return None

def _key_func(ps, dt, mid, output_token):
    """
    Assigns a unique code to each unique output and records it in the variable dictionary.

    Args:
        ps: Parameters object.
        dt: Data object containing the variable dictionary and file paths.
        mid: Matrix id.
        output: The generated output.

    Returns:
        int: The assigned code for the output.
    """
    output = ps.tokenizer.decode(output_token)
    output = ''.join([i if ord(i)<128 else 'ord({})'.format(ord(i)) for i in output])  # remove ascii
    if output == dt.std:
        key = "std"
    elif output in dt.var2k_dict:
        key = dt.var2k_dict[output]
    else:
        key = len(dt.var2k_dict.keys())
        dt.var2k_dict[output] = key
        out = "{}:{}\n{}\n%%%%%%\n".format(mid,key,output)
        with open(dt.path_code_file, 'a') as output_file:
            output_file.write(out)
    return key

def main():
    print('Process {}'.format(os.getpid()))
    mid = sys.argv[1]
    path_config = "parameters_config.json"
    if len(sys.argv)>1:
        path_config = sys.argv[2]
    ps = Parameters(path_config)
    dt = Data(ps)
    if check_done(dt, mid):
        return None
    model_load_func(ps, dt)
    standard_output_func(ps, dt)
    loc1, loc2 = file_func(ps, dt, mid)
    one_mid_func(ps, dt, mid, loc1, loc2)
    del ps.model
    return None

if __name__ == "__main__":
    main()
