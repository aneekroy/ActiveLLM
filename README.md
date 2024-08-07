# ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios

Active learning is designed to minimize annotation efforts by prioritizing instances that most enhance learning. However, many active learning strategies struggle with a 'cold start' problem, needing substantial initial data to be effective. This limitation often reduces their utility for pre-trained models, which already perform well in few-shot scenarios. To address this, we introduce ActiveLLM, a novel active learning approach that leverages large language models such as GPT-4, Llama 3, and Mistral Large for selecting instances. 

We demonstrate that ActiveLLM significantly enhances the classification performance of BERT classifiers in few-shot scenarios, outperforming both traditional active learning methods and the few-shot learning method SetFit. Additionally, ActiveLLM can be extended to non-few-shot scenarios, allowing for iterative selections. In this way, ActiveLLM can even help other active learning strategies to overcome their cold start problem. Our results suggest that ActiveLLM offers a promising solution for improving model performance across various learning setups.

## Repository Overview

This repository contains the experimental code and results (including prompts and answers) used for the ActiveLLM experiments as described in the paper by Markus Bayer and Christian Reuter.

Notice: Since the paper is only pre-printed, the code is not fully optimized and may contain some bugs.

### Directory Structure

- `/prompts/` - Directory where generated prompts are saved.
- `/logs/run/` - Directory where experiment results are stored.

### Scripts

1. **prompt_generation.py**
    - A script for generating the prompt for the LLM.
    
2. **model_training.py**
    - A script for training a BERT model based on the given answers of the LLM.

3. **automatic_run.py**
    - A script to run `model_training.py` multiple times with different run parameters.

### Scripts Details

#### prompt_generation.py
This script is used to generate prompts that will be fed into the large language models (LLMs) for instance selection. 
    - Includes the following parameters in the script:
      - `MODE = "10"`
      - `RUN = "1"`
      - `TASK = "cti"`
      - `EXAMPLES = 25`
      - `CONTINOUS = "IDXRECAP"`  # Options: False, NORECAP, RECAP, IDXRECAP

    - Modes:
      - Mode 1: CoT - it reiterates the advice
      - Mode 2: No CoT
      - Mode 3: No CoT but tasked to explain each instance
      - Mode 4: No advice + but CoT
      - Mode 5: Mode 4 and Mode 2
      - Mode 6: Mode 4 and Mode 3
      - Mode 7: Best one (Mode 4) with Guidelines
      - Mode 8: Mode 4 with 50 instances
      - Mode 9: Mode 4 with 100 instances
      - Mode 10: Mode 4 with 200 instances
      - Mode 11: Mode 4 with 400 instances

Usage:
```
python prompt_generation.py
```

#### model_training.py
This script trains a BERT model based on the instances selected by the LLM. The model is trained in a few-shot learning setup to evaluate the performance enhancement brought by ActiveLLM.

Usage:
```
python model_training.py --task <task_name> --mode <mode_name> --run <run_number> --modelclass <model_class> --activelearning <active_learning_strategy> --warmstart <warm_start_option>
```

#### automatic_run.py
This script automates the execution of `model_training.py` with various parameters to facilitate extensive experimentation.

Content:
```
# run model_training.py multiple times with different run parameters
import subprocess

TASK = ["sst2_25_cont_index_recap_1", "sst2_25_cont_index_recap_2", "sst2_25_cont_index_recap_3", "sst2_25_cont_index_recap_4", "sst2_25_cont_index_recap_5", "sst2_25_cont_index_recap_6", "sst2_25_cont_index_recap_7", "sst2_25_cont_index_recap_8", "sst2_25_cont_index_recap_9", "sst2_25_cont_index_recap_10", "sst2_25_cont_index_recap_11", "sst2_25_cont_index_recap_12"] 
MODES = ["smalltext_warmstart"]
RUNS = ["1"]
MODELCLASS = "default"
ACTIVELEARNING = ["PredictionEntropy"]
WARMSTART = "True"

for task in TASK:
    for mode in MODES:
        for activelearning in ACTIVELEARNING:
            for run in RUNS:
                subprocess.call(["python", "model_training.py", "--task", task, "--mode", mode, "--run", run, "--modelclass", MODELCLASS, "--activelearning", activelearning, "--warmstart", WARMSTART])
```

## Getting Started

To run the scripts, ensure you have all necessary dependencies installed. You can install the required packages using:


## Citation

If you use this code in your research, please cite our paper:

```
@misc{bayer2024activellmlargelanguagemodelbased,
      title={ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios}, 
      author={Markus Bayer and Christian Reuter},
      year={2024},
      eprint={2405.10808},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.10808}, 
}
```
## Contact

For any questions or issues, please contact [Markus Bayer](mailto:markus.bayer@example.com).
