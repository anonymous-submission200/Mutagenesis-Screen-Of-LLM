## Run example
```bash
# Activate environment. GPU is needed for running code
conda activate MgScreen3.10

# python Mmlu21_Experiment.py matrix_id address_JSON_configuration_file
python Mmlu21_Experiment.py model.layers.12.self_attn.v_proj.weight  path/to/your/parameters_config.json
```
---
## Guide to Writing `parameters_config.json`

This configuration file contains the parameters needed to run the program. Below is an explanation of each key and its expected value:

## Parameters

### `"model_address"`
- **Description**: The file path to the model you are using.
- **Example**: `"path/to/your/model"`

### `"task"`
- **Description**: The specific MMLU task name for testing.
- **Example**: `"international_law"`
  
### `"path_mmlu"`
- **Description**: The directory containing the MMLU benchmark dataset. This folder should include subfolders for `dev`, `test`, and `val` that contain MMLU csv files.
- **Example**: `"path/to/MMLU"`

### `"path_test_right"`
- **Description**: The path template for the MMLU dataset where the model provides correct answers with the random seed **YOU ARE USING** (random seed = 0 provided for Llama2-7b/Zephyr).
- **Example**: 
  - For Llama2-7b: `"path/to/for_llama2-7b/test_right"`
  - For Zephyr: `"path/to/for_zephyr/test_right"`

### `"path_test_wrong"`
- **Description**: The path template for the MMLU dataset where the model provides incorrect answers with the random seed **YOU ARE USING** (random seed = 0 provided for Llama2-7b/Zephyr).
- **Example**: 
  - For Llama2-7b: `"path/to/for_llama2-7b/test_wrong"`
  - For Zephyr: `"path/to/for_zephyr/test_wrong"`

### `"path_output_base"`
- **Description**: The directory where output files will be saved.
- **Example**: `"path/to/output/folder"`

### `"time"`
- **Description**: A timestamp label to append to output file names for easier identification.
- **Example**: `"110724"`

### `"max_test"`
- **Description**: The total number of MMLU questions to be used in testing.
- **Example**: `21` (as used in this paper)

### `"max_right"`
- **Description**: The number of MMLU questions where the model provides correct answers.
- **Example**: `14` (as used in this paper)

### `"step"`
- **Description**: The side length of the mutation square for testing.
- **Example**: `64` (as used in this paper)

### `"max_token"`
- **Description**: The maximum number of tokens allowed.
- **Example**: `4096` (as used in this paper)

### `"seed"`
- **Description**: Random seed value.
- **Example**: `0`

### `"pad_list"`
- **Description**: A list specifying the mutation types to apply. Supported values are `"max"` for maximum mutation, `"min"` for minimum mutation, and `"zero"` for zero mutation.
- **Example**: `["max", "min", "zero"]`
---

## Example `parameters_config.json`

```json
{
  "model_address": "path/to/your/model",
  "task": "international_law",
  "path_mmlu": "path/to/MMLU",
  "path_test_right": "path/to/for_llama2-7b/test_right",
  "path_test_wrong": "path/to/for_llama2-7b/test_wrong",
  "path_output_base": "path/to/output/folder",
  "time": "110724",
  "max_test": 21,
  "max_right": 14,
  "step": 64,
  "max_token": 4096,
  "seed": 0,
  "pad_list": ["max", "min", "zero"]
}
