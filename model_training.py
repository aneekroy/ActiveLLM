import pandas as pd
import numpy as np
import transformers
from datasets import load_dataset
import torch
from modelclass import ModelClass
from setfit_modelclass import SetFitModelClass
from small_text_modelclass import SmallTextModelClass
import os
import logging
import datetime
import random

os.environ["WANDB_DISABLED"] = "true"
dir = os.path.dirname(os.path.abspath(__file__))
import argparse
from small_text import (
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    LeastConfidence,
    PredictionEntropy,
    BALD,
    GreedyCoreset,
    ContrastiveActiveLearning,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cti")
parser.add_argument("--mode", type=str, default="baseline")
parser.add_argument("--run", type=str, default="1")
parser.add_argument("--modelclass", type=str, default="default")
parser.add_argument("--activelearning", type=str, default="None")
parser.add_argument("--warmstart", type=str, default=False)
args = parser.parse_args()

TASK = args.task
if "mnli" in TASK:
    EVAL = "_" + TASK.split("_")[1]
    TASK = "mnli"
else:
    EVAL = ""

MODE = args.mode
RUN = args.run
MODELCLASS = args.modelclass
ACTIVELEARNING = args.activelearning
WARMSTART = args.warmstart == "True"


# TASK = "qqp"
# MODE = "baseline"
# RUN = "5"

if "sst2_" in TASK or "cti" in TASK or "rte_" in TASK:
    EXAMPLES = int(TASK.split("_")[1])

    if len(TASK.split("_")) >= 3:
        CONTINOUS = True
        CONT_ITER = int(TASK.split("_")[-1])

        TASK = "_".join(TASK.split("_")[0:-1])
        print(TASK)
    else:
        CONTINOUS = False
        CONT_ITER = 1
elif "_" in TASK and TASK.split("_")[-1].isnumeric():
    EXAMPLES = int(TASK.split("_")[-1])
    CONT_ITER = 1
else:
    EXAMPLES = 32
    CONT_ITER = 1

if MODELCLASS == "default":
    logging_path = os.path.join(
        dir,
        "logs/run/" + TASK + EVAL + "/" + MODE + "/",
        "run_"
        + RUN
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".log",
    )
elif MODELCLASS == "setfit":
    logging_path = os.path.join(
        dir,
        "logs/run/fewshot/" + TASK + EVAL + "/" + MODE + "/",
        "run_"
        + RUN
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".log",
    )
elif MODELCLASS == "smalltext":
    logging_path = os.path.join(
        dir,
        "logs/run/smalltext/" + TASK + EVAL + "/" + MODE + "/" + ACTIVELEARNING + "/",
        "run_"
        + RUN
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".log",
    )

os.makedirs(os.path.dirname(logging_path), exist_ok=True)
# setting up logger to log to file in logs folder with name created from current time and date
logging.basicConfig(
    filename=logging_path,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def read_dataset(shuffled=False):
    if shuffled == False:
        df_train_path = os.path.join("data", TASK, "df_train_full.csv")
    else:
        print("Reading shuffled dataset")
        df_train_path = os.path.join(
            "data",
            TASK,
            "df_train_full_shuffled_" + str(shuffled) + ".csv",
        )

    if os.path.isfile(df_train_path):
        df_train = pd.read_csv(df_train_path)
    elif TASK == "gsm8k_main":
        dataset = load_dataset("gsm8k", "main")
        df_train = dataset["train"].to_pandas()
        df_train = df_train.rename(columns={"question": "text", "answer": "label"})
        df_train["index"] = df_train.index
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        os.makedirs(os.path.dirname(df_train_path), exist_ok=True)
        df_train.to_csv(df_train_path, index=False)
    else:
        raise FileNotFoundError(f"Dataset file {df_train_path} not found")

    # load test data
    if (
        "cti" in TASK
    ):  # even when shuffeled cti (e.g. cti_10) the test set is in cti folder
        df_test = pd.read_csv("data/cti/df_test.csv")
    elif TASK == "ag_news":
        df_test = load_dataset("SetFit/" + TASK, split="test").to_pandas()
    elif (
        "sst5" in TASK
        or "amazon_counterfactual_en" in TASK
        or "CR" in TASK
        or "emotion" in TASK
        or "ag_news" in TASK
    ):
        df_test = load_dataset(
            "SetFit/" + "_".join(TASK.split("_")[:-1]), split="test"
        ).to_pandas()
    elif TASK == "gsm8k_main":
        df_test = load_dataset("gsm8k", "main", split="test").to_pandas()
        df_test = df_test.rename(columns={"question": "text", "answer": "label"})
    else:
        if "sst2" in TASK:
            df_test = load_dataset(
                "glue", "sst2", split="validation" + EVAL
            ).to_pandas()
        elif "rte" in TASK:
            df_test = load_dataset("glue", "rte", split="validation" + EVAL).to_pandas()
        else:
            df_test = load_dataset("glue", TASK, split="validation" + EVAL).to_pandas()
        if "sst2" in TASK or TASK == "cola":
            df_test = df_test.rename(columns={"sentence": "text", "label": "label"})
        elif TASK == "qqp":
            df_test = df_test.rename(
                columns={"question1": "text1", "question2": "text2"}
            )
        elif "rte" in TASK:
            df_test = df_test.rename(
                columns={"sentence1": "text1", "sentence2": "text2"}
            )
        elif TASK == "qnli":
            df_test = df_test.rename(columns={"question": "text1", "sentence": "text2"})
        elif TASK == "mnli" or TASK == "ax":
            df_test = df_test.rename(
                columns={"premise": "text1", "hypothesis": "text2"}
            )
        elif TASK == "mrpc" or TASK == "stsb" or TASK == "wnli":
            df_test = df_test.rename(
                columns={"sentence1": "text1", "sentence2": "text2"}
            )
        # add index
        df_test["index"] = df_test.index
    return df_train, df_test


# read active learning instances
if MODE == "baseline":
    actively_learned_instances = list(range(0, EXAMPLES * CONT_ITER))
elif ACTIVELEARNING == "random_filtering":
    actively_learned_instances = []
else:

    # if it is smalltext evaluation
    if "smalltext_coldstart" in MODE or "smalltext_warmstart" in MODE:
        with open(
            os.path.join(
                dir,
                "answers/"
                + TASK
                + "/"
                + MODE
                + "/"
                + ACTIVELEARNING
                + "/list_"
                + RUN
                + ".txt",
            ),
            "r",
        ) as f:
            actively_learned_instances = f.read()
            actively_learned_instances = actively_learned_instances.split(",")
            actively_learned_instances = actively_learned_instances[
                : EXAMPLES * CONT_ITER
            ]
            actively_learned_instances = [int(x) for x in actively_learned_instances]
            print(actively_learned_instances)

            assert len(actively_learned_instances) == EXAMPLES * CONT_ITER
        # in answers\TASK\MODE\list_RUN.txt
    else:
        if MODELCLASS == "smalltext" and WARMSTART == False:
            actively_learned_instances = []
        else:
            with open(
                os.path.join(
                    dir, "answers/" + TASK + "/" + MODE + "/list_" + RUN + ".txt"
                ),
                "r",
            ) as f:
                actively_learned_instances = f.read()
                actively_learned_instances = actively_learned_instances.split(",")
                actively_learned_instances = actively_learned_instances[
                    : EXAMPLES * CONT_ITER
                ]
                actively_learned_instances = [
                    int(x) for x in actively_learned_instances
                ]

                assert len(actively_learned_instances) == EXAMPLES * CONT_ITER


df_train, df_test = read_dataset(shuffled=RUN)

if ACTIVELEARNING == "random_filtering":
    actively_learned_instances = random.sample(
        range(len(df_train)), EXAMPLES * CONT_ITER
    )

if (
    TASK == "qqp"
    or TASK == "rte"
    or TASK == "qnli"
    or TASK == "mnli"
    or TASK == "mrpc"
    or TASK == "ax"
    or TASK == "stsb"
    or TASK == "wnli"
):
    x_test = (df_test["text1"].tolist(), df_test["text2"].tolist())
else:
    x_test = df_test["text"].tolist()

# if stsb get labels as floats
if TASK == "stsb":
    y_test = [round(x, 2) for x in df_test["label"].astype(float).tolist()]
else:
    y_test = df_test["label"].tolist()

if (
    MODELCLASS == "smalltext"
):  # work-around for having all instances as possible active learning instances for smalltext
    labeled_indices = actively_learned_instances
    actively_learned_instances = list(range(0, len(df_train)))

x_train, y_train = [], []
for instance_index in actively_learned_instances:
    if (
        TASK == "qqp"
        or TASK == "rte"
        or TASK == "qnli"
        or TASK == "mnli"
        or TASK == "mrpc"
        or TASK == "ax"
        or TASK == "stsb"
        or TASK == "wnli"
    ):
        x_train.append(
            (df_train["text1"][instance_index], df_train["text2"][instance_index])
        )
    else:
        x_train.append(df_train["text"][instance_index])
    if TASK == "stsb":
        y_train.append(round(float(df_train["label"][instance_index]), 2))
    else:
        y_train.append(df_train["label"][instance_index])

# if qqp, convert list of tuples to tuple of lists
if (
    TASK == "qqp"
    or TASK == "rte"
    or TASK == "qnli"
    or TASK == "mnli"
    or TASK == "mrpc"
    or TASK == "ax"
    or TASK == "stsb"
    or TASK == "wnli"
):
    x_train = ([x[0] for x in x_train], [x[1] for x in x_train])


if MODELCLASS == "smalltext":
    model = SmallTextModelClass(
        "bert-base-uncased", False, None, len(set(y_train)), seed=1
    )

    active_train = model.preprocess_data(model.tokenizer, x_train, y_train)
    # active_test = preprocess_data(self.tokenizer, x_dev, y_dev)

    # Active learner
    if ACTIVELEARNING == "LeastConfidence":
        query_strategy = LeastConfidence()
    elif ACTIVELEARNING == "PredictionEntropy":
        query_strategy = PredictionEntropy()
    elif ACTIVELEARNING == "BALD":
        query_strategy = BALD()
    elif ACTIVELEARNING == "GreedyCoreset":
        query_strategy = GreedyCoreset()
    elif ACTIVELEARNING == "ContrastiveActiveLearning":
        query_strategy = ContrastiveActiveLearning()
    else:
        query_strategy = LeastConfidence()
    active_learner = PoolBasedActiveLearner(
        model.clf_factory, query_strategy, active_train
    )

    if WARMSTART:
        ACTIVE_SAMPLES = 300  # when starting with 25_cont_4, we have 100 ActiveGPT samples and want 300 samples in the end
        QUERY_SAMPLES = 25
        labeled_indices = model.initialize_active_learner(
            active_learner, active_train.y, labeled_indices
        )
    else:
        if TASK == "ag_news":
            ACTIVE_SAMPLES = EXAMPLES * CONT_ITER
            QUERY_SAMPLES = 5
        else:
            ACTIVE_SAMPLES = 300
            QUERY_SAMPLES = 25

        labeled_indices = range(0, QUERY_SAMPLES)
        labeled_indices = model.initialize_active_learner(
            active_learner, active_train.y, labeled_indices
        )

    try:
        labeled_indices = model.perform_active_learning(
            active_learner,
            active_train,
            labeled_indices,
            active_samples=ACTIVE_SAMPLES,
            query_samples=QUERY_SAMPLES,
        )
        logging.info("Labeled indices: " + str(labeled_indices))
    except PoolExhaustedException:
        print("Error! Not enough samples left to handle the query.")
    except EmptyPoolException:
        print("Error! No more samples left. (Unlabeled pool is empty)")

    assert len(labeled_indices) == ACTIVE_SAMPLES

    # get actively learned instances
    x_train = [x_train[i] for i in labeled_indices]
    y_train = [y_train[i] for i in labeled_indices]


if TASK == "cti":
    seed_list = [42, 109, 27, 158, 77]
else:
    seed_list = [42, 109, 27, 158, 77]
accuracy_list, f1_list, corr_list = [], [], []
for seed in seed_list:
    if TASK == "stsb":
        num_labels = 1
    else:
        num_labels = len(set(y_train))

    if MODELCLASS == "default" or MODELCLASS == "smalltext":
        model = ModelClass("bert-base-uncased", False, None, num_labels, seed=seed)
        if TASK == "cti":
            train_bs = 8
            test_bs = 32
        else:
            train_bs = 128
            test_bs = 512
        evaluations = model.train(x_train, y_train, x_test, y_test, train_bs, test_bs)
    elif MODELCLASS == "setfit":
        model = SetFitModelClass(seed=seed)
        evaluations = model.train(x_train, y_train, x_test, y_test)

    if TASK == "ax" or TASK == "stsb":
        corr_list.append(evaluations["eval_Correlation: "])
    elif TASK == "cola":
        corr_list.append(evaluations["eval_Correlation: "])
        accuracy_list.append(evaluations["eval_Accuracy: "])
        f1_list.append(evaluations["eval_F1: "])
    else:
        accuracy_list.append(evaluations["eval_Accuracy: "])
        f1_list.append(evaluations["eval_F1: "])


if TASK == "ax" or TASK == "stsb":
    logging.info("Corr:" + str(corr_list))
    logging.info("Corr:" + str(sum(corr_list) / len(seed_list)))
    print("Corr:", corr_list)
    print("Corr:", sum(corr_list) / len(seed_list))
elif TASK == "cola":
    logging.info("Corr:" + str(corr_list))
    logging.info("Corr:" + str(sum(corr_list) / len(seed_list)))
    logging.info("Acc:" + str(accuracy_list))
    logging.info("Acc:" + str(sum(accuracy_list) / len(seed_list)))
    logging.info("F1:" + str(f1_list))
    logging.info("F1:" + str(sum(f1_list) / len(seed_list)))

    print("Corr:", corr_list)
    print("Corr:", sum(corr_list) / len(seed_list))
    print(accuracy_list)
    print(sum(accuracy_list) / len(seed_list))
    print(f1_list)
    print(sum(f1_list) / len(seed_list))

else:
    logging.info("Acc:" + str(accuracy_list))
    logging.info("Acc:" + str(sum(accuracy_list) / len(seed_list)))
    logging.info("F1:" + str(f1_list))
    logging.info("F1:" + str(sum(f1_list) / len(seed_list)))

    print(accuracy_list)
    print(sum(accuracy_list) / len(seed_list))
    print(f1_list)
    print(sum(f1_list) / len(seed_list))
