#the prompt and code with 5 in-context learning examples from the https://github.com/FranxYao/chain-of-thought-hub was used

import os
import shutil
import sys
import json

import pandas as pd
import GPUtil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Parameters:
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
        self.task = config["tast"]
        self.seed = config["seed"]
        self.model_address = config["model_address"]
        self.path_mmlu = config["path_mmlu"]
        self.path_test_right = os.path.join(config["path_test_right"], f"{task}_test.csv")
        self.path_test_wrong = os.path.join(config["path_test_wrong"], f"{task}_test.csv")
        self.path_output = os.path.join(config["path_output_base"], f"{task}_{seed}")

        # Other configurations
        self.max_test = config["max_test"]
        self.max_right = config["max_right"]
        self.step = config["step"]
        self.max_token = config["max_token"]
        self.pad_list = config["pad_list"]
        self.time = config["time"]

        # Dynamically generate development set path and prompt template
        self.path_dev = os.path.join(self.path_mmlu, f"dev/{task}_dev.csv")
        self.raw_prompt = f"The following are multiple-choice questions (with answers) about {task.replace('_', ' ')}.\n\n"
        
        # Verify paths and raise errors if necessary
        self._verify_paths()

    def _verify_paths(self):
        """
        Verify the existence of required directories and files. Raise errors if any validation fails.
        """
        # Check if the MMLU folder exists
        if not os.path.isdir(self.path_mmlu):
            raise FileNotFoundError(f"The MMLU folder does not exist: {self.path_mmlu}")

        # Check if the test right file exists
        if not os.path.isfile(self.path_test_right):
            raise FileNotFoundError(f"The test-right file does not exist: {self.path_test_right}")

        # Check if the test wrong file exists
        if not os.path.isfile(self.path_test_wrong):
            raise FileNotFoundError(f"The test-wrong file does not exist: {self.path_test_wrong}")

        # Check if the development set file exists
        if not os.path.isfile(self.path_dev):
            raise FileNotFoundError(f"The development set file does not exist: {self.path_dev}")

        # Check if the output directory exists or can be created
        if not os.path.exists(self.path_output):
            try:
                os.makedirs(self.path_output)
            except Exception as e:
                raise RuntimeError(f"Failed to create the output directory: {self.path_output}\nError: {e}")
            
class Data:
    def __init__(self, ps):
        """
        Initialize the Data class.

        Args:
            ps: Parameters object containing paths and configuration settings.
        """
        # File paths
        # Standard file: Records output from the standard model with no mutation
        self.path_std_file = os.path.join(ps.path_output, f'mmlu_{ps.task}_{ps.seed}_std_{ps.time}.txt')

        # Code file: Records the unique integer code assigned to each unique output (phenotype) from mutations
        self.path_code_file = os.path.join(ps.path_output, f'mmlu_{ps.task}_{ps.seed}_{{}}_code_{ps.time}.txt')  # {{mid}}

        # Log file: Records each non-silent mutation in the format {pad} {mid}:{loc1},{loc2}:{code of phenotype}
        self.path_log_file = os.path.join(ps.path_output, f'mmlu_{ps.task}_{ps.seed}_{{}}_log_{ps.time}.txt')  # {{mid}}

        # Finish file: Records the site (loc1, loc2) of the last mutation calculated; (loc1,loc2) mutation location on matrix
        self.path_finished_file = os.path.join(ps.path_output, f'mmlu_{ps.task}_{ps.seed}_{{}}_finished_{ps.time}.txt')  # {{mid}}

        # Done file: Records mid if the calculation for the mid matrix is complete
        self.path_done_file = os.path.join(ps.path_output, 'mmlu_done_{ps.time}.txt')

        # Load data
        self.dev_df = self.load_csv(ps.path_dev, nrows=5)
        self.test_right_df, self.right_nmb = self.load_test_data(ps.path_test_right, max_rows=ps.max_right)
        self.test_wrong_df, self.wrong_nmb = self.load_test_data(ps.path_test_wrong, max_rows=ps.max_test - self.right_nmb)
        self.total_nmb = self.right_nmb + self.wrong_nmb

        # Initialize other attributes
        self.input_token_list = []
        self.right_answer_list = []
        self.path_file_dict = {}
        self.pad_dict = {}
        self.var_dict = {}
        self.choice = ["A", "B", "C", "D"]

    def load_csv(self, path, nrows=None):
        """
        Load a CSV file.

        Args:
            path (str): Path to the CSV file.
            nrows (int): Maximum number of rows to load. Load all rows if None.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            return pd.read_csv(path, header=None, nrows=nrows)
        except FileNotFoundError:
            raise FileNotFoundError(f"[Data] File not found: {path}")
        except Exception as e:
            raise RuntimeError(f"[Data] Error loading CSV file {path}: {e}")

    def load_test_data(self, path, max_rows):
        """
        Load test data, limiting the maximum number of rows.

        Args:
            path (str): Path to the test data file.
            max_rows (int): Maximum number of rows to load.

        Returns:
            tuple: (pd.DataFrame, int) The loaded DataFrame and the actual number of rows.
        """
        df = self.load_csv(path)
        if df.shape[0] > max_rows:
            df = df[:max_rows]
        return df, df.shape[0]

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
    ps.model = AutoModelForCausalLM.from_pretrained(ps.model_address).to(f"cuda:{dt.gpu_id}")
    ps.model.generation_config.pad_token_id = ps.tokenizer.pad_token_id
    ps.para_dict = ps.model.state_dict()
    return None

def input_func(ps, dt):
    """
    Generate input prompts and corresponding input tokens from the development set, 
    while also generating the correct answers list.

    Args:
        ps: Parameters object containing configurations and tokenizer.
        dt: Data object containing test datasets and other required data.
    """
    # Initialize prompts for the development set
    _dev_prompt_func(ps, dt)

    # Clear the input token list and correct answers list
    dt.input_token_list = []
    dt.right_answer_list = []

    # Combine correct and incorrect test datasets
    combined_df = pd.concat([dt.test_right_df, dt.test_wrong_df], ignore_index=True)

    # Process each row to generate prompts and encode input tokens
    def process_row(row):
        # Generate the correct answer
        dt.right_answer_list.append(row.iloc[-1])
        # Generate input prompt and encode it
        prompt = _one_prompt_func(ps, dt, row)
        input_token = ps.tokenizer.encode(prompt, return_tensors="pt", padding=True).to(f"cuda:{dt.gpu_id}")
        dt.input_token_list.append(input_token)

    # Apply the processing function to each row
    combined_df.apply(process_row, axis=1)

    return None

def standard_output_func(ps, dt):
    """
    Collect the answers from the standard model (non-mutated state).

    Args:
        ps: Parameters object containing model configurations.
        dt: Data object containing input and output paths.
    """
    dt.path_std_file = dt.path_std_file.format(ps.task, ps.seed)
    if os.path.isfile(dt.path_std_file):
        # If standard output already exists, read it
        with open(dt.path_std_file, 'r') as std_file:
            dt.std = std_file.read()
    else:
        # Generate standard output
        std_list = _one_round_output_func(ps, dt)
        dt.std = '| |'.join(std_list)
        with open(dt.path_std_file, 'w') as std_file:
            std_file.write(dt.std)
    return None

def file_func(ps, dt, mid):
    """
    Manage record files used to store mutation calculation information and results.
    Make a code dictionary of processed mutations on mid matrix.
    Extract the last processed mutation location.

    Args:
        ps: Parameters object containing configuration settings.
        dt: Data object containing file paths and processing states.
        mid: Matrix id.

    Returns:
        tuple: The location (loc1, loc2) of mutation to start.
    """
    # Log file: Append a marker for tracking the last processed location
    dt.path_log_file = dt.path_log_file.format(ps.task, ps.seed, mid)
    if os.path.isfile(dt.path_log_file):
        with open(dt.path_log_file, 'a') as output_file:
            # Add a marker to indicate the point of breakdown from the last session
            output_file.write("%%%%%%\n")

    # Code file: Record or retrieve mutation codes
    dt.path_code_file = dt.path_code_file.format(ps.task, ps.seed, mid)
    if not os.path.isfile(dt.path_code_file):
        # If the code file does not exist, initialize it
        with open(dt.path_code_file, 'w') as output_file:
            standard_output_func(ps, dt)
            output = ("seed:{} step: {}\nstandard output:\n{}\n%%%%%%\n".format(ps.seed, ps.step, dt.std))
            output_file.write(output)
    else:
        # Extract existing codes
        with open(dt.path_code_file, 'r') as raw_file:
            temp_list = raw_file.read().split('\n%%%%%%\n')
            std = temp_list[0]
            std = '\n'.join(std.split('\n')[2:])  # Remove the first 2 lines
            temp_list.pop(0)
            temp_list.pop(-1)
            if temp_list:
                _var_dict_func(ps, dt, temp_list)
            if std != dt.std:
                x = input("Standard input differs! Need to redo the whole set!")

    # Finish file: Retrieve the site of the last calculated mutation
    loc1, loc2 = [0, 0]
    dt.path_finished_file = dt.path_finished_file.format(ps.task, ps.seed, mid)
    if os.path.isfile(dt.path_finished_file):
        with open(dt.path_finished_file, 'r') as raw_file:
            loc1, loc2 = [int(item) for item in raw_file.readline().split(',')]
            loc1 += ps.step  # Advance to the next step
    return loc1, loc2

def one_mid_func(ps, dt, mid, loc1, loc2):
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


def _one_prompt_func(ps, dt, smp_df):
    """
    Generate a single input prompt by combining the development set prompt with the sample-specific prompt.

    Args:
        ps: Parameters object containing configuration and tokenizer.
        dt: Data object containing development set and sample data.
        smp_df: A single sample from the dataset.

    Returns:
        str: The complete input prompt.
    """
    smp_prompt = _format_sample(ps, dt, smp_df, include_answer=False)
    prompt = ps.dev_prompt + smp_prompt

    # Ensure the prompt length does not exceed the maximum token limit
    while len(ps.tokenizer.tokenize(prompt)) + 1 > ps.max_token:
        prompt_split = prompt.split("\n\n")
        prompt_split.pop(1)
        prompt = '\n\n'.join(prompt_split)

    return prompt


def _dev_prompt_func(ps, dt):
    """
    Generate the development set prompt.

    Args:
        ps: Parameters object containing configuration and tokenizer.
        dt: Data object containing the development set.
    """
    ps.dev_prompt = ps.raw_prompt
    for i in range(dt.dev_df.shape[0]):
        ps.dev_prompt += _format_sample(ps, dt, dt.dev_df.iloc[i])
    return None


def _format_sample(ps, dt, smp_df, include_answer=True):
    """
    Format a single sample into the required text format.

    Args:
        ps: Parameters object.
        dt: Data object containing choices.
        smp_df: A single sample from the dataset.
        include_answer (bool): Whether to include the correct answer.

    Returns:
        str: The formatted sample as a string.
    """
    frmt_txt = smp_df[0]
    for i in range(smp_df.shape[0] - 2):
        frmt_txt += '\n{}. {}'.format(dt.choice[i], smp_df.iloc[i + 1])
    frmt_txt += "\nAnswer:"
    if include_answer:
        frmt_txt += " {}\n\n".format(smp_df.iloc[-1])
    return frmt_txt


def _get_gpu():
    """
    Identify the GPU with the most available memory.

    Returns:
        int: The ID of the GPU with the largest free memory.
    """
    GPUs = GPUtil.getGPUs()
    return sorted(GPUs, key=lambda x: x.memoryFree, reverse=True)[0].id

def _one_round_output_func(ps, dt):
    """
    Executes one round of inference and generates an output list.

    Args:
        ps: Parameters object containing model and tokenizer configurations.
        dt: Data object containing input tokens and correct answers.

    Returns:
        list: A list of inference results and accuracy statistics.
    """
    # Perform inference and generate answers
    output_list = _generate_outputs(ps, dt)

    # Calculate accuracy statistics
    stats = _calculate_accuracy(output_list, dt)

    # Append statistics to the output list
    output_list.append(
        f"right: {stats['right_in_right']}|{stats['right_in_wrong']} "
        f"wrong: {stats['wrong_in_right']}|{stats['wrong_in_wrong']} "
        f"total: {stats['total_correct']}|{stats['total_wrong']}|{dt.total_nmb}"
    )
    return output_list


def _generate_outputs(ps, dt):
    """
    Generates an answer list based on input tokens.

    Args:
        ps: Parameters object containing model and tokenizer configurations.
        dt: Data object containing input tokens and correct answers.

    Returns:
        list: A list of generated answers.
    """
    output_list = []
    torch.manual_seed(ps.seed)  # Set random seed for reproducibility

    for input_token in dt.input_token_list:
        # Generate output tokens
        output_token = ps.model.generate(input_token, max_new_tokens=1)
        last_token = output_token[0, -1]
        # Decode the last token and replace newlines with a placeholder
        answer = ps.tokenizer.decode(last_token).replace('\n', '^n')
        output_list.append(answer)

    return output_list


def _calculate_accuracy(output_list, dt):
    """
    Calculates accuracy and statistical information from the output list.

    Args:
        output_list (list): List of answers generated by the model.
        dt: Data object containing correct answers and other statistics.

    Returns:
        dict: A dictionary containing accuracy statistics.
    """
    stats = {
        "right_in_right": 0,  # Correct answers from correct test cases
        "right_in_wrong": 0,  # Correct answers from incorrect test cases
        "wrong_in_right": dt.right_nmb,  # Initial value: all correct test cases
        "wrong_in_wrong": dt.wrong_nmb,  # Initial value: all incorrect test cases
        "total_correct": 0,  # Total correct answers
        "total_wrong": 0,  # Total wrong answers
    }

    for idx, answer in enumerate(output_list):
        right_answer = dt.right_answer_list[idx]

        if right_answer == answer:
            stats["total_correct"] += 1

            if idx < dt.right_nmb:  # Correct test cases
                stats["right_in_right"] += 1
                stats["wrong_in_right"] -= 1
            else:  # Incorrect test cases
                stats["right_in_wrong"] += 1
                stats["wrong_in_wrong"] -= 1
        else:
            stats["total_wrong"] += 1

    return stats


def _var_dict_func(ps, dt, temp_list):
    """
    Extract codes recorded in the code file and populate the variable dictionary.

    Args:
        ps: Parameters object.
        dt: Data object containing the variable2code dictionary.
        temp_list: List of data from the code file.
    """
    for item in temp_list:
        key = '\n'.join(item.split('\n')[1:])
        code = item.split('\n')[0].split(':')[1]
        dt.var_dict[key] = code
    return None


def _finished_func(ps, dt, loc1, loc2):
    """
    Records the site of the most recently processed mutation.

    Args:
        ps: Parameters object.
        dt: Data object containing file paths.
        loc1 (int): Starting loc1 site.
        loc2 (int): Starting loc2 site.
    """
    with open(dt.path_finished_file, 'w') as output_file:
        output_file.write("{},{}".format(loc1, loc2))
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
    output_list = _one_round_output_func(ps, dt)
    output = '| |'.join(output_list)
    # modify non-ASCII characters
    output = ''.join([i if ord(i) < 128 else f'ord{ord(i)}' for i in output])

    if output != dt.std:
        code = _code_func(ps, dt, mid, output)
        # Log the mutation details
        with open(dt.path_log_file, 'a') as log_file:
            log_file.write(f"{pad} {mid}:{loc1},{loc2}:{code}\n")
        with open(dt.path_file_dict[pad], 'a') as output_file:
            out = f"{mid}:{loc1},{loc2}:{code}\n"
            output_file.write(out)
    return None


def _code_func(ps, dt, mid, output):
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
    if output in dt.var_dict:
        code = dt.var_dict[output]
    else:
        code = len(dt.var_dict.keys())
        dt.var_dict[output] = code
        out = f"{mid}:{code}\n{output}\n%%%%%%\n"
        with open(dt.path_code_file, 'a') as output_file:
            output_file.write(out)
    return code

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
    input_func(ps, dt)
    standard_output_func(ps, dt)
    loc1, loc2 = file_func(ps, dt, mid)
    one_mid_func(ps, dt, mid, loc1, loc2)
    del ps.model
    return None

if __name__ == "__main__":
    main()
