# Guide to Writing `parameters_config.json`

This configuration file contains the parameters needed to run the program. Below is an explanation of each key and its expected value:

## Parameters

### `"model_address"`
- **Description**: The file path to the model you are using.
- **Example**: `"path/to/your/model"`

### `"path_output"`
- **Description**: The directory where output files will be saved.
- **Example**: `"path/to/output/folder"`

### `"input_text"`
- **Description**: The input text to model.
- **Inputs used in the paper**: 
* Drosophila Experiment: `"The life cycle of Drosophila:"`
* Python0 and Python10 Experiment: `"The following is a python program for bubble sort:"`
* Java Experiment: `"The following is a Java program for bubble sort:"`
* Newton Experiment: `"The scientific accomplishments and influences of Isaac Newton:"`
* P53 Experiment: `"Tell me 10 different signal pathways through which p53 is involved in cancer development:"`

### `"temperature"`
- **Description**: Temperature for model.
- **Example**: `0.7` (as used in this paper)

### `"step"`
- **Description**: The side length of the mutation square for testing.
- **Example**: `64` (as used in this paper)

### `"max_length"`
- **Description**: The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens..
- **Example**: `150` (as used for most experiments in this paper except for Newton P53 experiments which used 300)

### `"seed"`
- **Description**: Random seed value.
- **Example**: `0`

### `"pad_list"`
- **Description**: A list specifying the mutation types to apply. Supported values are `"max"` for maximum mutation, `"min"` for minimum mutation, and `"zero"` for zero mutation.
- **Example**: `["max", "min", "zero"]`

### `"experiment_name"`
- **Description**: Experiment name to append to output file names for easier identification.
- **Example**: `zephyr_Drosophila`

### `"time"`
- **Description**: A timestamp label to append to output file names for easier identification.
- **Example**: `"110724"`
---

## Example `parameters_config.json`

```json
{
  "model_address": "path/to/your/model",
  "path_output": "path/to/output/folder",
  "experiment_name": "zephyr_Drosophila",
  "input_text": "The life cycle of Drosophila:",
  "max_length": 150,
  "temperature": 0.7,
  "step": 64,
  "seed": 0,
  "time": "110724",
  "pad_list": ["max", "min", "zero"]
}

