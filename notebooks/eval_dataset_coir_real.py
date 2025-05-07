import os
import time
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch
from mteb.tasks.Retrieval import SyntheticText2SQLRetrieval
import random

# Check if GPU is available, otherwise switch to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, switching to CPU")

print("Using device:", device)

# Define the sentence-transformers model name
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr1e-05-epochs10-zeropadding_noepsilon_temperature200_full_test_sanitycheck"
#model_name = "distilbert-base-uncased_ED-fever-lr2e-05-epochs10-temperature20_full_dev_full_test"
model_name = "distilbert-base-uncased"
model_name = "Alibaba-NLP/gte-modernbert-base"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs10-temperature200_full_test"
#model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
start_time = time.time()  # Record the start time

# Load the model
# model = SentenceTransformer(os.path.join(os.getcwd(), "beir\\examples\\retrieval\\training\\output\\distilbert-base-uncased_CosSim-fever-lr2e-05-epochs10-temperature20_full_dev"))
# model = SentenceTransformer(model_name)
model = SentenceTransformer(os.path.join(os.getcwd(), r"output\Alibaba-NLP\gte-modernbert-base-synthetictext2sql-lr1e-5-epochs10\bckp_checkpoint-5625"))
# model = SentenceTransformer(os.path.join(os.getcwd(), r"output\Alibaba-NLP\gte-modernbert-base-synthetictext2sql-lr1e-5-epochs10_test\checkpoint-1"))
#model = SentenceTransformer(model_name)
model = model.to(device)  # Ensure the model is loaded to the correct device

# Load the HotpotQA task from MTEB
task = SyntheticText2SQLRetrieval()
task.load_data()

# Print the total number of queries in the full HotpotQA test set
total_queries_test = len(task.queries["test"])
print(f"Total number of queries in the full Text2SQL test set: {total_queries_test}")

# Print the total number of documents in the full HotpotQA test set
total_documents_test = len(task.corpus["test"])
print(f"Total number of documents in the full Text2SQL test set: {total_documents_test}")

# Ensure the entire test set is used for evaluation
# No subset is created in this version; we use the full test set

# Create an MTEB evaluation instance with the full task
evaluation = MTEB(tasks=[task])

# Run the evaluation
# results = evaluation.run(model, verbosity=2, eval_splits=["test"], output_folder=f"results\\{model_name}")
results = evaluation.run(model, verbosity=2, eval_splits=["test"], output_folder=f"results\\{model_name}_trained_new_ed_idk")

end_time = time.time()  # Record the end time

# Calculate the duration
duration_seconds = end_time - start_time
hours = int(duration_seconds // 3600)
minutes = int((duration_seconds % 3600) // 60)
seconds = int(duration_seconds % 60)

print(f"Model evaluation took: {hours} hours, {minutes} minutes, and {seconds} seconds.")

