import glob
import os
import json
import pathlib
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import TrainerCallback
from datasets import Dataset, load_dataset
from collections import defaultdict
from beir import util
from beir.datasets.data_loader import GenericDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.current_device())  # Prints the current device index


# Define paths
model_name = "Alibaba-NLP/gte-modernbert-base"
#model_name = "snowflake-arctic-embed-m-v1.5_CosSim"
model = SentenceTransformer(model_name)
save_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-synthetictext2sql-lr1e-5-epochs10_test")
os.makedirs(save_dir, exist_ok=True)

# Load dataset
VALIDATION_SIZE = 0.1
queries_ds = load_dataset("CoIR-Retrieval/synthetic-text2sql", name="queries", split="queries")
corpus_ds = load_dataset("CoIR-Retrieval/synthetic-text2sql", name="corpus", split="corpus")
qrels_ds = load_dataset("CoIR-Retrieval/synthetic-text2sql", name="default", split="train")
train_size = len(qrels_ds)
# qrels_ds.shuffle(seed=42)
new_train_size = int(train_size * (1 - VALIDATION_SIZE))
new_train_data = qrels_ds.select(range(new_train_size))
validation_data = qrels_ds.select(range(new_train_size, train_size))
queries = {q["_id"]: q["text"] for q in queries_ds}
corpus = {c["_id"]: {"text": c["text"], "title": ""} for c in corpus_ds}

dev_corpus_ids = set(item["corpus-id"] for item in validation_data)
dev_query_ids = set(item["query-id"] for item in validation_data)
dev_corpus = {c["_id"]: {"text": c["text"], "title": ""} for c in corpus_ds if c["_id"] in dev_corpus_ids}
dev_queries = {q["_id"]: q["text"] for q in queries_ds if q["_id"] in dev_corpus_ids}
dev_queries = {q["_id"]: q["text"] for q in queries_ds if q["_id"] in dev_query_ids}

qrels_train = defaultdict(dict)
for entry in new_train_data:
    qid = entry["query-id"]
    cid = entry["corpus-id"]
    qrels_train[qid][cid] = entry.get("score", 1)

qrels_val = defaultdict(dict)
for entry in validation_data:
    qid = entry["query-id"]
    cid = entry["corpus-id"]
    qrels_val[qid][cid] = entry.get("score", 1)


train_data = [{"query": queries[qid], "positive": corpus[pos_id]["text"]} for qid, pos_doc_ids in qrels_train.items() for pos_id in pos_doc_ids]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
dev_data = [{"doc_id": doc_id, "text": doc["text"]} for doc_id, doc in dev_corpus.items()]
dev_dataset = Dataset.from_list(dev_data)

# # Convert BEIR data into Hugging Face Dataset format
# train_data = [{"query": queries[qid], "positive": corpus[pos_id]["text"]} for qid, pos_doc_ids in qrels.items() for pos_id in pos_doc_ids]
# train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
# dev_data = [{"doc_id": doc_id, "text": doc["text"]} for doc_id, doc in val_corpus.items()]
# dev_dataset = Dataset.from_list(dev_data)  # Convert to Hugging Face Dataset


# Define evaluator
ir_evaluator = InformationRetrievalEvaluator(
    queries=dev_queries, corpus=dev_corpus, relevant_docs=validation_data,
    name="text2sql-dev", show_progress_bar=True
)

model.to(device)

# ** Custom Callback to Save Best Model ** WE DONT NEED THIS TO SAVE THE BEST MODEL JUST UNCOMMENT THE TRAINING ARGUMENTS THAT ARE COMMENTED OUT
class BestModelCallback(TrainerCallback):
    def __init__(self, evaluator, save_path, metric="ndcg_at_10"):
        self.evaluator = evaluator
        self.save_path = save_path
        self.best_score = -float("inf")
        self.metric = metric

    def on_epoch_end(self, args, state, control, **kwargs):
        results = self.evaluator.compute_metrics(state.model)
        current_score = results.get(self.metric, -float("inf"))

        if current_score > self.best_score:
            self.best_score = current_score
            state.model.save(self.save_path)
            print(f"New best model saved with {self.metric} = {current_score:.4f}")


# ** Training Arguments ** Uncomment arguments to use ir_evaluator after each epoch
training_args = SentenceTransformerTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    # learning_rate=1e-5,
    learning_rate=1e-3,
    warmup_steps=int(len(train_dataset) * 10 / 16 * 0.1),
    logging_steps=1, 
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    # evaluation_strategy="no",
    save_total_limit=2,
    #load_best_model_at_end=True,
    #metric_for_best_model="eval_hotpotqa-dev_energy_distance_ndcg@10",
    #greater_is_better=True,
    max_steps=1
)

# ** Resume Trainer from Last Epoch ** set eval dataset to use ir_evaluator and uncomment evaluator argument
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    loss=losses.MultipleNegativesRankingLoss(model=model),
    #evaluator=ir_evaluator,
    callbacks=[]
)

# Train the model using the trainer
trainer.train()
