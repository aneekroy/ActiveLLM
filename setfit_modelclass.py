from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
import numpy as np
import logging
from datasets import Dataset
from datasets import load_dataset
import os
dir = os.path.dirname(os.path.abspath(__file__))
from transformers import set_seed
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer


def compute_metrics(y_pred, y_test):
    # make differentiation between binary and multi-class classification
    predictions, labels = y_pred, y_test
    if len(np.unique(labels)) == 2:
        return {
            "eval_Accuracy: " : accuracy_score(labels, predictions),
            "eval_F1: " : f1_score(labels, predictions, pos_label=1), 
            "Precision_1: " : recall_score(labels, predictions, pos_label=1),
            "Recall_1: " : precision_score(labels, predictions, pos_label=1),
            "Precision_0: " : recall_score(labels, predictions, pos_label=0),
            "Recall_0: " : precision_score(labels, predictions, pos_label=0),
            # methew's correlation coefficient
            "eval_Corr: ": matthews_corrcoef(labels, predictions),
        }
    elif len(np.unique(labels)) >= 5:
        predictions = np.squeeze(predictions)
        return { #spearman correlation
            "eval_Accuracy: " : accuracy_score(labels, predictions),
            "eval_Corr: ": spearmanr(labels, predictions)[0],
        }
    else:
        return {
            "eval_Accuracy: " : accuracy_score(labels, predictions),
            "eval_F1: " : f1_score(labels, predictions, average="macro"), 
            "Precision: " : recall_score(labels, predictions, average="macro"),
            "Recall: " : precision_score(labels, predictions, average="macro"),
        }

def set_seed(seed=109):
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)

# Class representing a model
class SetFitModelClass:
    def __init__(self, seed=109):
        # init a huggingface auto model based on the model name
        set_seed(seed)
        self.model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2").to("cuda")
        #self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    def predict(self, texts, bs = 128):
        predictions = self.predictor(texts, bs).tolist()
        return np.argmax(predictions, axis=1)

    def predictor(self, texts, bs = 128):
        if isinstance(texts, str):
            texts = [texts]

        predictions = []
        with torch.no_grad():
            self.model.eval()
            if type(texts) is not tuple:
                break_len = len(texts)
            else:
                break_len = len(texts[0])

            for i in range(0, break_len, bs):
                if type(texts) is not tuple:
                    batch_x = self.tokenizer(texts[i:i+bs], padding=True, truncation=True, return_tensors="pt").to("cuda")
                else:
                    batch_x = self.tokenizer(texts[0][i:i+bs], texts[1][i:i+bs], padding=True, truncation=True, return_tensors="pt").to("cuda")
                predictions.extend(torch.nn.functional.softmax(self.model(**batch_x)[0], dim=1).cpu().detach().tolist())
        return np.array(predictions)

    def train(self, x_train, y_train, x_dev, y_dev, train_bs = 16, test_bs = 32):
        # transform x_train, y_train, x_dev, y_dev to lists of dictionaries where each dictionary represents a data instance
        # if x_train and x_dev are tuples, then the dictionaries have two keys: "text1" and "text2"
        # if x_train and x_dev are not tuples, then the dictionaries have one key: "text"
        if type(x_train) is not tuple:
            train_data = {"text": x_train, "label": y_train}
            test_data = {"text": x_dev, "label": y_dev}
        else:
            train_data = {"text1": x_train[0], "text2": x_train[1], "label": y_train}
            test_data = {"text1": x_dev[0], "text2": x_dev[1], "label": y_dev}
        
        # convert to datasets
        train_ds = Dataset.from_dict(train_data)
        test_ds = Dataset.from_dict(test_data)
        

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            #loss_class=CosineSimilarityLoss,
            metric=compute_metrics,
            batch_size=train_bs,
            num_epochs=4,
            #num_iterations=20, # Number of text pairs to generate for contrastive learning
            #num_epochs=1 # Number of epochs to use for contrastive learning
            #column_mapping={"sentence": "text", "label": "label"}
        )

        trainer.train()
        # logging the results
        logging.info("Training results:")
        evaluations = trainer.evaluate()

        logging.info(evaluations)

        return evaluations

    # def evaluate(self, x_test, y_test):
    #     if type(x_test) is not tuple:
    #         test_encodings = self.tokenizer(x_test, padding=True, truncation=True, return_tensors="pt")
    #     else:
    #         test_encodings = self.tokenizer(*x_test, padding=True, truncation=True, return_tensors="pt")

    #     test_dataset = Dataset(test_encodings, y_test)
    #     trainer = Trainer(
    #         model=self.model,
    #         compute_metrics=compute_metrics,
    #     )
    #     logging.info("Test results:")
    #     logging.info(trainer.evaluate(test_dataset))


    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path, num_labels):
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
