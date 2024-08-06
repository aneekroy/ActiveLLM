# run model_training.py multiple times with different run parameters
import subprocess

TASK = ["rte"]
MODES = ["llama_3_100"]
RUNS = ["1"]
MODELCLASS = "default"
ACTIVELEARNING = ["None"]
WARMSTART = ""

# TASK = ["sst2_25_cont_index_recap_2"] # continous sampling naming: sst2_25_cont_index_recap_4 -> 25 samples, start with 4th iteration (after 100 instances)
# MODES = ["10"]
# RUNS = ["1"]
# MODELCLASS = "smalltext"
# ACTIVELEARNING = ["PredictionEntropy"]
# WARMSTART = "True"

for task in TASK:
    for mode in MODES:
        for activelearning in ACTIVELEARNING:
            for run in RUNS:
                subprocess.call(["python", "model_training.py", "--task", task, "--mode", mode, "--run", run, "--modelclass", MODELCLASS, "--activelearning", activelearning, "--warmstart", WARMSTART])
                #subprocess.call(["python", "calc_avg.py", "--task", task, "--mode", mode])


# Classic Active Learning
# TASK = ["sst2_25_cont_index_recap_1"] 
# MODES = ["10"]
# RUNS = ["1"]
# MODELCLASS = "smalltext"
# ACTIVELEARNING = ["PredictionEntropy"]
# WARMSTART = "True"
         
# Just doing Classic Active Learning (continous) - ONLY LABLES AS OUTPUT (no eval)
# Wenn mit ActiveGPT warmstart: sst2_25_cont_index_recap_4 und WARMSTART == "True"  
# TASK = ["sst2_25_cont_index_recap_4"] # continous sampling naming: sst2_25_cont_index_recap_4 -> 25 samples, start with 4th iteration (after 100 instances)
# MODES = ["10"]
# RUNS = ["1"]
# MODELCLASS = "smalltext"
# ACTIVELEARNING = ["PredictionEntropy"]
# WARMSTART = "True"
                
# cont Active Learning evaluation (25 eval - 50 eval - 75 eval - 100 eval ..) - ALSO MODES = ["smalltext_cold/warmstart"] and MODELCLASS = "default"
# ! FIRST: copy answers of classic active learning to answers folder 
# -> from "\logs\run\smalltext\cti_25_cont_index_recap\10\ACTIVELEARNING" into script "answers\ActiveLearningLabels_to_List.py"
# -> set warm or cold start in script and make commas with alt+strg
# -> and output to "\answers\cti_25_cont_index_recap\smalltext_coldstart\ACTIVELEARNING\list_1.txt"
# TASK = ["sst2_25_cont_index_recap_1", "sst2_25_cont_index_recap_2", "sst2_25_cont_index_recap_3", "sst2_25_cont_index_recap_4", "sst2_25_cont_index_recap_5", "sst2_25_cont_index_recap_6", "sst2_25_cont_index_recap_7", "sst2_25_cont_index_recap_8", "sst2_25_cont_index_recap_9", "sst2_25_cont_index_recap_10", "sst2_25_cont_index_recap_11", "sst2_25_cont_index_recap_12"] 
# MODES = ["smalltext_coldstart"]
# RUNS = ["1"]
# MODELCLASS = "default"
# ACTIVELEARNING = ["PredictionEntropy"]
# WARMSTART = "False"/"True"

# Evaluation of selection sizes
# TASK = ["cti_10", "cti_20", "cti_30", "cti_40", "cti_50", "cti_60", "cti_70", "cti_80", "cti_90", "cti_100"] 
# MODES = ["baseline"]
# RUNS = ["1"]
# MODELCLASS = "default"
# ACTIVELEARNING = [""]
# WARMSTART = ""