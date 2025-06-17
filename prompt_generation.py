import tiktoken
import pandas as pd
import re
from datasets import load_dataset
import pathlib
import datetime

MODE = "10"
RUN = "1"
TASK = "cti"
EXAMPLES = 25
CONTINOUS = "IDXRECAP"  # False, NORECAP, RECAP, IDXRECAP

if MODE == "10" or MODE == "12":
    PROMPT_EXAMPLES = 200

if CONTINOUS == "NORECAP":  # the last actively selected instances are not included
    PATH_ADDITION = "_cont_no_recap"
elif CONTINOUS == "RECAP":  # the last actively selected instances are included
    PATH_ADDITION = "_cont"
elif (
    CONTINOUS == "IDXRECAP"
):  # the last actively selected instances are included by index
    PATH_ADDITION = "_cont_index_recap"
else:
    PATH_ADDITION = ""


def clean(instance, match):
    # remove RT
    text_processed = instance.replace("RT ", "")
    # remove URLs
    text_processed = match.sub(r"", text_processed)

    return text_processed


def is_duplicate(instances, instance, match):
    cleaned_instance = clean(instance, match)

    for prompt_instance in instances:
        if clean(prompt_instance, match) == cleaned_instance:
            return True

    return False


def instance_prompt_generation(dataset, token_limit, prompt_token_count, enc):
    # Regex for removing URLs in the duplicate detection function
    match = re.compile(
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        re.MULTILINE | re.UNICODE,
    )

    prompt_instances = ""
    prompt_instances_list = []
    prompt_examples = PROMPT_EXAMPLES

    if TASK == "qqp":
        # merge text1 and text2
        dataset["text"] = (
            "Question 1: " + dataset["text1"] + " \n Question 2: " + dataset["text2"]
        )
    if TASK == "rte" or TASK == "mrpc" or TASK == "stsb" or TASK == "wnli":
        # merge text1 and text2
        dataset["text"] = (
            "Sentence 1: " + dataset["text1"] + " \n Sentence 2: " + dataset["text2"]
        )
    if TASK == "qnli":
        # merge text1 and text2
        dataset["text"] = (
            "Question: " + dataset["text1"] + " \n Sentence: " + dataset["text2"]
        )
    if TASK == "mnli" or TASK == "ax":
        # merge text1 and text2
        dataset["text"] = (
            "Premise: " + dataset["text1"] + " \n Hypothesis: " + dataset["text2"]
        )

    # if continuous mode, read the last actively selected instances
    if CONTINOUS != "False":
        recap_instance_path = pathlib.Path(
            "answers/"
            + TASK
            + "_"
            + str(EXAMPLES)
            + PATH_ADDITION
            + "/"
            + MODE
            + "/list_"
            + RUN
            + ".txt"
        )
        if recap_instance_path.is_file():
            with open(recap_instance_path) as f:
                recap_instances = f.read()
                recap_instances = recap_instances.split(",")
                recap_instances = [int(i) for i in recap_instances]
            # get max index
            last_instance = max(recap_instances) + 1
        else:
            last_instance = 0
            recap_instances = []
    else:
        last_instance = 0
        recap_instances = []

    if CONTINOUS == "RECAP":
        # if recap mode is full include all instances textually (so that gpt can read which instances were labeled)
        # replace the indices with the text of the instances
        recap_instances = dataset["text"][recap_instances].tolist()
    elif CONTINOUS == "IDXRECAP":
        # more prompt instances need to be included
        prompt_examples = last_instance + PROMPT_EXAMPLES
        last_instance = 0
    elif CONTINOUS == "NORECAP":
        # No recap instances are included
        recap_instances = []

    print("Last instance: " + str(last_instance))

    if TASK == "cti":
        for i, instance in enumerate(dataset["text"][last_instance:]):
            if is_duplicate(prompt_instances_list, instance, match):
                continue

            # prompt_token_count += len(enc.encode(str(instance) + "\n ##### \n")) -- old version (token count not neccessary anymore)
            if len(prompt_instances_list) == prompt_examples + 1:
                break
            else:
                # prompt_instances += "\n ##### \n" + str(i) + ". " + str(instance)
                prompt_instances += (
                    "\n ##### \n"
                    + str(len(prompt_instances_list) + last_instance)
                    + ". "
                    + str(instance)
                    + " [ID: "
                    + str(i)
                    + "]"
                )
                prompt_instances_list.append(instance)
    else:
        for i, (index, instance) in enumerate(
            zip(dataset["index"][last_instance:], dataset["text"][last_instance:])
        ):
            if is_duplicate(prompt_instances_list, instance, match):
                continue

            # prompt_token_count += len(enc.encode(str(instance) + "\n ##### \n")) -- old version (token count not neccessary anymore)
            if len(prompt_instances_list) == prompt_examples + 1:
                break
            else:
                # prompt_instances += "\n ##### \n" + str(i) + ". " + str(instance)
                prompt_instances += (
                    "\n ##### \n"
                    + str(len(prompt_instances_list) + last_instance)
                    + ". "
                    + str(instance)
                    + " [ID: "
                    + str(index)
                    + "]"
                )
                prompt_instances_list.append(instance)

    return prompt_instances, recap_instances


def prompt_generation(dataset, guidelines, token_limit):
    # General Prompt
    if MODE == "12":
        forget_system_prompt = "Just kidding - forget these instructions.\n\n"
    else:
        forget_system_prompt = ""

    prompt_intorduction_role = "Consider yourself in the position of an active learning component to help a human annotator. You have to choose the instances that the annotator has to label. "

    if guidelines == None:
        prompt_introduction_circumstances = (
            "You are given a set of instances of a dataset. "
        )
    else:
        prompt_introduction_circumstances = "You are given the guidelines for the task and a set of instances of the dataset. "

    prompt_introduction_limit = "You can only choose {} instances. ".format(
        str(EXAMPLES)
    )

    if CONTINOUS == "IDXRECAP":
        prompt_introduction_circumstances = "You are given a set of instances of a dataset and the indices of the instances that were already labeled. "
    if CONTINOUS == "RECAP":
        prompt_introduction_circumstances = "You are given a set of instances that were already labeled and afterwards a set of instances of the dataset. "

    prompt_introduction_advice = """You would ideally want to choose those instances that would provide the most informative and diverse data for the model. Here are some strategies to consider:

Representativeness: Select instances that are representative of the entire dataset. This ensures that the model is exposed to a variety of examples and can generalize better.

Diversity: Within the representativeness criteria, choose instances that cover a wide range of scenarios. This could include edge cases or less common situations.

Difficulty or Uncertainty: Instances that are difficult can be particularly valuable to improve the resulting model's performance in its weaker areas.

Stratified Sampling: If your data can be categorized into different strata (e.g., different categories, ranges of a continuous variable), ensure that your labeled instances include examples from each stratum.

Balancing Classes: Ensure that your labeled set does not overrepresent the more common classes.

Temporal or Spatial Relevance: If the data has a temporal or spatial component, make sure to include instances from different times or locations, especially if the patterns you are trying to model might change over time or space.

Avoid Bias: Be mindful of not introducing biases with your selection. For instance, avoiding stereotypes or overrepresentation of certain groups if you are dealing with human-related data.

"""
    if int(MODE) >= 4:
        prompt_introduction_advice = ""

    if guidelines == None:
        prompt_introduction_guidelines = ""
    else:
        prompt_introduction_guidelines = (
            "\n\nLabel Guidelines for the human annotator:\n'''"
            + guidelines
            + "'''\n\n"
        )

    # Output formatting

    if (
        MODE == "1"
        or MODE == "4"
        or MODE == "7"
        or MODE == "8"
        or MODE == "10"
        or MODE == "12"
    ):
        prompt_output_formatting = "Please think step by step about what you would do to select the instances to label. After this provide the list of instances that you would label, separated by a comma. For example, if you would label the instances 1, 4, 5 then the output should be: 1, 4, 5 "
    elif MODE == "2" or MODE == "5":
        prompt_output_formatting = "The output format should be a list of the instances that you would label, separated by a comma. For example, if you would label the instances 1, 4, 5 then the output should be: 1, 4, 5 "
    elif MODE == "3" or MODE == "6":
        prompt_output_formatting = "Please describe instance by instance why you would select or not select it for labelling. Do not stop before you successfully found {} that you would suggest to label. After this provide the list of the instances that you would label, separated by a comma. For example, if you would label the instances 1, 4, 5 then the output should be: 1, 4, 5. ".format(
            str(EXAMPLES)
        )

    # Instance presentation

    prompt_instance_presentation = (
        'The following instances are given to you (seperated with "\\n ##### \\n"): \n'
    )

    if CONTINOUS == "IDXRECAP":
        prompt_recap = "\n\nThe following instances have already been labeled (indices of the dataset above): "
    elif CONTINOUS == "RECAP":
        prompt_recap = '\n\nThe following instances have already been labeled (seperated with "\\n ##### \\n"): '
    else:
        prompt_recap = ""

    initial_prompt = (
        forget_system_prompt
        + prompt_intorduction_role
        + prompt_introduction_circumstances
        + prompt_introduction_limit
        + prompt_introduction_advice
        + prompt_introduction_guidelines
        + prompt_output_formatting
    )

    enc = tiktoken.encoding_for_model("gpt-4")
    initial_prompt_length = len(enc.encode(initial_prompt))

    prompt_instances, recap_instances = instance_prompt_generation(
        dataset, token_limit, initial_prompt_length, enc
    )

    if len(recap_instances) == 0:
        if guidelines == None:
            prompt_introduction_circumstances = (
                "You are given a set of instances of a dataset. "
            )
        else:
            prompt_introduction_circumstances = "You are given the guidelines for the task and a set of instances of the dataset. "

        initial_prompt = (
            forget_system_prompt
            + prompt_intorduction_role
            + prompt_introduction_circumstances
            + prompt_introduction_limit
            + prompt_introduction_advice
            + prompt_introduction_guidelines
            + prompt_output_formatting
        )

        final_prompt = initial_prompt + prompt_instance_presentation + prompt_instances
    else:
        if CONTINOUS == "IDXRECAP":
            final_prompt = (
                initial_prompt
                + prompt_instance_presentation
                + prompt_instances
                + prompt_recap
                + str(recap_instances).replace("[", "").replace("]", "")
            )
        else:
            final_prompt = (
                initial_prompt
                + prompt_recap
                + "\n\n"
                + "\n ##### \n".join(recap_instances)
                + "\n\n"
                + prompt_instance_presentation
                + prompt_instances
            )

    return final_prompt


def read_dataset(shuffled=False):
    # check if task dataset already exists
    if EXAMPLES != 32:
        if CONTINOUS == False:
            pathlib.Path("data/" + TASK + "_" + str(EXAMPLES)).mkdir(
                parents=True, exist_ok=True
            )
        else:
            pathlib.Path("data/" + TASK + "_" + str(EXAMPLES) + PATH_ADDITION).mkdir(
                parents=True, exist_ok=True
            )
    else:
        pathlib.Path("data/" + TASK).mkdir(parents=True, exist_ok=True)
    if shuffled == False:
        print("Not Implemented!")
        # if pathlib.Path("data/" + TASK + "/df_train_full.csv").is_file():
        #     df_train = pd.read_csv("data/" + TASK + "/df_train_full.csv")
        # else:
        #     # get task datasets from huggingface
        #     dataset = load_dataset("glue", TASK)
        #     # convert to pandas dataframe
        #     df_train = dataset["train"].to_pandas()

        #     if TASK == "sst2" or TASK == "cola":
        #         # rename columns
        #         df_train = df_train.rename(
        #             columns={"sentence": "text", "label": "label"}
        #         )
        #     if TASK == "qqp":
        #         # rename columns question1 to text1 and question2 to text2
        #         df_train = df_train.rename(
        #             columns={
        #                 "question1": "text1",
        #                 "question2": "text2",
        #                 "label": "label",
        #             }
        #         )
        #     if TASK == "rte" or TASK == "mrpc" or TASK == "stsb" or TASK == "wnli":
        #         # rename columns question1 to text1 and question2 to text2
        #         df_train = df_train.rename(
        #             columns={
        #                 "sentence1": "text1",
        #                 "sentence2": "text2",
        #                 "label": "label",
        #             }
        #         )
        #     if TASK == "qnli":
        #         # rename columns question1 to text1 and question2 to text2
        #         df_train = df_train.rename(
        #             columns={
        #                 "question": "text1",
        #                 "sentence": "text2",
        #                 "label": "label",
        #             }
        #         )
        #     if TASK == "mnli" or TASK == "ax":
        #         # rename columns question1 to text1 and question2 to text2
        #         df_train = df_train.rename(
        #             columns={
        #                 "premise": "text1",
        #                 "hypothesis": "text2",
        #                 "label": "label",
        #             }
        #         )

        #     # add index
        #     df_train["index"] = df_train.index
        #     # save dataset
        #     df_train.to_csv("data/" + TASK + "/df_train_full.csv", index=False)
    else:
        if (EXAMPLES == 32 or EXAMPLES == 25) and pathlib.Path(
            "data/" + TASK + "/df_train_full_shuffled_" + str(shuffled) + ".csv"
        ).is_file():
            df_train = pd.read_csv(
                "data/" + TASK + "/df_train_full_shuffled_" + str(shuffled) + ".csv"
            )
        else:
            if TASK == "cti":
                df_train = pd.read_csv(
                    "data/" + TASK + "/df_train_full_shuffled_" + str(shuffled) + ".csv"
                )
            elif TASK in [
                "sst2",
                "cola",
                "qqp",
                "rte",
                "mrpc",
                "stsb",
                "wnli",
                "qnli",
                "mnli",
                "ax",
            ]:
                # get task datasets from huggingface
                dataset = load_dataset("glue", TASK)
                # convert to pandas dataframe
                df_train = dataset["train"].to_pandas()
                # rename columns

                if TASK == "sst2" or TASK == "cola":
                    # rename columns
                    df_train = df_train.rename(
                        columns={"sentence": "text", "label": "label"}
                    )
                if TASK == "qqp":
                    # rename columns question1 to text1 and question2 to text2
                    df_train = df_train.rename(
                        columns={
                            "question1": "text1",
                            "question2": "text2",
                            "label": "label",
                        }
                    )
                if TASK == "rte" or TASK == "mrpc" or TASK == "stsb" or TASK == "wnli":
                    # rename columns question1 to text1 and question2 to text2
                    df_train = df_train.rename(
                        columns={
                            "sentence1": "text1",
                            "sentence2": "text2",
                            "label": "label",
                        }
                    )
                if TASK == "qnli":
                    # rename columns question1 to text1 and question2 to text2
                    df_train = df_train.rename(
                        columns={
                            "question": "text1",
                            "sentence": "text2",
                            "label": "label",
                        }
                    )
                if TASK == "mnli" or TASK == "ax":
                    # rename columns question1 to text1 and question2 to text2
                    df_train = df_train.rename(
                        columns={
                            "premise": "text1",
                            "hypothesis": "text2",
                            "label": "label",
                        }
                    )
            elif TASK == "gsm8k_main":
                dataset = load_dataset("gsm8k", "main")
                df_train = dataset["train"].to_pandas()
                df_train = df_train.rename(
                    columns={"question": "text", "answer": "label"}
                )
            else:
                dataset = load_dataset("SetFit/" + TASK)
                # convert to pandas dataframe
                df_train = dataset["train"].to_pandas()

            # add index
            df_train["index"] = df_train.index
            # shuffle dataset
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            # save shuffled dataset
            if EXAMPLES != 32:
                df_train.to_csv(
                    "data/"
                    + TASK
                    + "_"
                    + str(EXAMPLES)
                    + PATH_ADDITION
                    + "/df_train_full_shuffled_"
                    + str(shuffled)
                    + ".csv",
                    index=False,
                )
            else:
                df_train.to_csv(
                    "data/"
                    + TASK
                    + "/df_train_full_shuffled_"
                    + str(shuffled)
                    + ".csv",
                    index=False,
                )

    return df_train


# df_train = read_dataset()
# shuffle dataset
# df_train_shuffled = df_train.sample(frac=1).reset_index(drop=False)
# save shuffled dataset
# df_train_shuffled.to_csv("data/df_train_full_shuffled_5.csv", index=False)

if MODE == "7":
    guidelines = """Put yourself in the role of a cybersecurity expert who has received initial information about a Microsoft Exchange incident. Since the information is relatively new, you want to gain further insight by looking at some Twitter posts. You search Twitter with some keywords regarding Microsoft Exchange. As you notice that many posts are not insightful you start to annotate the posts in terms of their relevance to you.\n\n#### Do not only consider the text but also the referenced websites.\n\n- - -\n\nWhat is labeled as **Relevant**?\n\n* Information that might be good to know for cybersecurity experts\n* Specific information about the vulnerabilites, how to perform them, how to mitigate them, what an attacker can do with exploiting the vulnarability ..\n* Expliticly mentioned numbers, methods, proof of concepts, code, fixes\n* Tweets that include IOCs (IPs, Hashwerte, usw.), vulnerabilies (ProxyLogon), malware names, exploits (DearCry), or CVE IDs can be very quickly flagged as relevant\n* Relevant information is more valuable than irrelevant information - if a post contains both relevant and irrelevant information, it is to be labeled as relevant\n    * For example "As of March 8, over 30K servers in the US have been hit by the recent Exchange #zero-day attack, which leaves behind a web shell that allows hackers to access the server to steal data & install malware" --> Relevant\n\n--\n\nWhat is labeled as **Not Relevant**?\n\n* Information that is too general\n* Information regarding the scope (which authorities, business branches and how many)\n    * This then includes posts about how many servers have not been patched yet, for example, or the following: „Microsoft Exchange Server attacks: 'They're being hacked faster than we can count', says security company“\n    * Likewise this refers to posts like „The European Banking Authority said it had been a victim of a cyberattack targeting its Microsoft Exchange Servers” --> Not relevant\n    * But on the other hand, a post like "RT Researchers have acquired a list of 86,000 IP addresses of MS Exchange servers infected worldwide by the mass compromises" is still relevant, as the IP list would be of interest.\n* If a post only links to a page without really providing information, it should be marked as not relevant, even if the linked page contains important information. Above all, a post should be marked as not relevant if it is not exactly clear what to expect on the page. Otherwise, when, for example, the text says that it is IOC info or a security advisory, you can consider it relevant (or check the page again). Please still consider the following examples:\n    * “GovInfoSecurity \| Analysis: Microsoft Exchange Server Hacks https://bit\.ly/2QgZb9P” --> not relevant \(too inaccurate\)\n    * “Chile's bank regulator shares IOCs after Microsoft Exchange hack https://ift.tt/3lrnfm3” --> Relevant (IOCs are very interesting for security experts)\n    * “Here's what we know so far about the massive Microsoft Exchange hack https://www.wxii12.com/article/here-s-what-we-know-so-far-about-the-massive-microsoft-exchange-hack/35793771?utm\_campaign=snd-autopilot” --> not Relevant (really no information in the text; too vague what to expect on the page)\n    * "RT How the Microsoft Exchange hack could impact your organization https://tek.io/2Oieqi5(https://t.co/un4YdQkbA3)" --> Not Relevant (too inaccurate)\n    * “CISA Updates Microsoft Exchange Advisory to Include China Chopper https://dlvr.it/RvhdjL” --> Relevant (You know what to expect on the site; besides, CISA is a major player)\n    * “At Least 10 Hacking Groups Are Exploiting Microsoft Exchange Server Flaws [PCMag] https://best.photography/articles/543893/at-least-10-hacking-groups-are-exploiting-microsoft-exchange-server-flaws/” --> Not Relevant (sounds relevant at first, because possibly the 10 hacking groups are mentioned and they are possibly known. However, the link is not trustworthy and also not callable) – relevant would have been fine here too, especially if the link was not strange.\n    * „It really was only a matter of time https://www.bleepingcomputer.com/news/security/new-dearcry-ransomware-is-targeting-microsoft-exchange-servers/” –-> Not Relevant (looks like an exciting link, but in the post there is no information about the link - you don't know what to expect) – relevant would also have been fine\n* Information that seems to be spam\n* Information that is highly politicly motivated\n* Information for general public (non experts)\n* Information that is speculative\n* "Casual news" for the general public\n* General cybersecurity advises (for the general public)\n* Podcasts, Interviews or personal opinions are to be marked as not relevant\n* Smaller service companies that report that their systems are updated and safe\n* News aggregations with several other news are not relevant\n    * "RT Read this week's digest to find out the latest updates in the #Microsoft Exchange vulnerabilities as well as how #hackers were able to breach 150,000 surveillance cameras from inside hospitals, jails and Tesla."\n\n**When in doubt --> Not Relevant** \n\n"""
else:
    guidelines = None

df_train = read_dataset(shuffled=RUN)
prompt = prompt_generation(df_train, guidelines, 25000)

# Insights
# Mode 1: CoT - it reiterates the advice
# Mode 2: No CoT ==> Explanation for each positive instance
# Mode 3: No CoT but tasked to explain each instance - has problems to explain the instances till 32 are found (often needs to be reminded in the response to find 32 instances)
# Mode 4: No advice + but CoT - ChatGPT writes its own advice directly in the response
# Mode 5: Mode 4 and Mode 2
# Mode 6: Mode 4 and Mode 3
# Mode 7: Best one (Mode 4) with Guidelines
# Mode 8: Mode 4 with 50 instances
# Mode 9: Mode 4 with 100 instances
# Mode 10: Mode 4 with 200 instances
# Mode 11: Mode 4 with 400 instances


# Most of the time GPT returns 32 instances

# GPT almost always proceeds from the first instance (meaning that giving it more instances might even decrease the performance??
# Or is it better as the attention can also look at the other instances?)

# 400 -> Given the large number of instances, I would not be able to read and analyze each one in this format. However, I would apply these principles to select the most informative and diverse instances.
# 200 -> best one (HOWEVER: In paper set it in relation to context length)

# IDXRECAP -> GPT often utilizes the analysis module
# IDXRECAP -> We expected to quickly have a very big prompt - however, while in the other scenarios the older, not labelled instances
# cannot be viewed by GPT - it now also chooses older instances (which might be very sensible)


# MIXTRAL -> versteht sehr häufig die Aufgabe nicht; muss teilweise daran erinnert werden; besonders lange Kontexte funktionieren nicht
# weswegen wir dann auch mit weniger Instanzen gearbeitet haben; Kann selten genau 32 Instanzen finden; Erklärt sich aber auch am Anfang
# jedoch nur kurz


# GEMINI -> very bad context length (worse than mixtral); Sometimes it says that it is unable to do this as a language model; struggels
# much more than the GPT4 model;


# GPT-3.5 (model=text-davinci-002-render-sha): is easier to handle than gemini (less refusal and more understanding of the task)
# or mixtral (understanding of text); however, context length is of course also a problem; gpt much better at selecting 32
# instances than gemini or mixtral


# MISTRAL Large -> very very similar to GPT-4 but faster! And much briefer explanation (advices to itself)


# Further Experiments/Discussions
# How good is GPT to label the instances?
# Schauen ob die Ergebnisse besser werden, wenn ich 100, 200 oder 300 Instanzen nehme
# Mit einer Task schauen wie viele label-Vorschläge zu Verbesserung führen (10, 20, 32, 50,.., 100)
# Description that the main task is the CTI task as the might be more unfamiliar with it
# Clear description of the prompt engeneering process (first instructions then the instances - causal attention)
# GLUE tasks


# VS
# Setfit
# Active Learning Methods - where the first 10 examples are used for overcoming the cold start problem


# How many epochs? For Diss 3-5 to be consistent with the other experiments?
# For the paper 100? -> https://arxiv.org/pdf/2006.04884.pdf, esp. https://arxiv.org/pdf/2012.15723.pdf

# save prompt

if EXAMPLES != 32:
    TASK = TASK + "_" + str(EXAMPLES)

if CONTINOUS != "False":
    TASK = TASK + PATH_ADDITION

pathlib.Path("prompts/" + TASK + "/" + MODE).mkdir(parents=True, exist_ok=True)

if CONTINOUS == "False":
    with open(
        "prompts/" + TASK + "/" + MODE + "/prompt_50000_" + RUN + ".txt",
        "w",
        encoding="UTF-8",
    ) as f:
        f.write(prompt)
else:
    # include timestamp
    with open(
        "prompts/"
        + TASK
        + "/"
        + MODE
        + "/prompt_50000_"
        + RUN
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt",
        "w",
        encoding="UTF-8",
    ) as f:
        f.write(prompt)
