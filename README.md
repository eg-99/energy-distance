# HPML Project: Energy Distance Project Training and Inference Instructions
## Team Information
- **Team Name**: IBM Project #8
- **Members**:
- Chandhru Karthick (ck3255)
- Chhavi Dixit (cd3496)
- Elie Gross (eg3346)
## 1. Problem Statment and Motivation
Part of long term project to explore better optimal retrieval metrics
* Traditionally Cosine Similarity used as distance metric for IR tasks
Potential Issues:
* Cosine requires a single vector (typically [CLS] to represent a seq) => Loss of information
Approach:
* Physics inspired energy-distance as an alternate metric
* Includes statistics of different token with multiple vectors for each query
* Treats query as a “cloud” ☁️ and the document as a “point” ⚫️
Value:
* Longer queries have more noise and are harder to represent with single embedding [CLS]
* Ergo: Tends to work better than cosine similarity for longer queries.
Ongoing experiments explore:
* Different distance metrics within energy-distance (L1, Hamming_L1…)
* Comparison with distance metrics like Jenson Shannon-Divergence
* Impact on IR tasks with different benchmark datasets (CoIR)


## 2. Model Description
Generally, for the Energy Distance project, the base model used is based on DistilBERT, a lighter and faster variant of BERT, pre-trained on the same corpus with knowledge distillation. The architecture follows the SentenceTransformer paradigm, combining a transformer encoder with pooling and normalization layers to generate fixed-size sentence embeddings.

For the CoIR Text2SQL benchmark, the model employed is based on Alibaba-NLP’s `gte-modernbert-base`, a BERT-based encoder optimized for general-purpose text embeddings across multiple languages and retrieval tasks. It follows the SentenceTransformer architecture, consisting of a BERT encoder followed by mean pooling and L2 normalization to generate fixed-length sentence embeddings.

* Framework: PyTorch, with HuggingFace Transformers and Sentence-Transformers
* Base Architecture: bert-base-uncased

  * 12 transformer layers

  * 12 attention heads

  * Hidden size: 768

  * Intermediate (FFN) size: 3072

  * Activation: GELU

* Custom Layers and Modifications:

  * Mean pooling over token embeddings

  * L2 normalization on the pooled sentence vector

* Similarity function: An energy distance (ED-L1) metric is used in place of cosine similarity to better model intra-query distributional structures.

This architecture is trained using Multiple Negatives Ranking Loss, scaled by a fixed factor (20.0), to enhance separation in the embedding space for information retrieval tasks.


## 3. Final Results Summary


| Model                     | ndcg@10 |
|---------------------------|---------|
| Leaderboard               | 0.6469  |
| Baseline benchmark        | 0.63429 |
| Single step               | 0.65781 |
| 10 Epochs, larger lr      | 0.45195 |
| 10 Epochs, smaller lr     | 0.64807 |


## 4. Reproducibility Instructions

### A. Insomnia Cluster Setup
Follow instructions in `bash_stuff.sh`

### Setting up Python Environment and Installing Required Libraries
1. conda create --name myenv39 python=3.9
2. conda activate myenv39
3. pip install --upgrade pip --index-url https://pypi.org/simple
4. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
5. git clone https://github.com/gnatesan/sentence-transformers-3.4.1.git
6. git clone https://github.com/gnatesan/mteb-1.34.14.git
7. pip install -e /path_to_sentence-transformers/sentence-transformers-3.4.1
8. pip install -e /path_to_mteb/mteb-1.34.14
9. git clone https://github.com/gnatesan/beir.git

### B. Sanity Check
1. conda create --name testenv python=3.9
2. conda activate testenv
3. pip install --upgrade pip --index-url https://pypi.org/simple
4. pip install sentence-transformers==3.4.1
5. pip install mteb==1.34.14
6. sbatch inference_CosSim.sh (Make sure the batch script calls eval_dataset.py and a baseline model is being used. *i.e. model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")*)
7. Cross reference the inference results with what is on the leaderboard. https://huggingface.co/spaces/mteb/leaderboard

### C. Model Training - New COIR Additions

// Connect to insomnia server with cuid user

srun --pty -t 0-04:00 --gres=gpu:2 -A edu /bin/bash

USERNAME=$(whoami)

cd /insomnia001/depts/edu/users/$USERNAME

// Set up the Python environment as noted below for myenv39

PATH="/insomnia001/depts/edu/users/$USERNAME/anaconda3/bin:/insomnia001/depts/edu/users/$USERNAME/anaconda3/sbin:$PATH"

CONDA_PYTHON_EXE=/insomnia001/depts/edu/users/$USERNAME/anaconda3/bin/python


// Modify paths and parameters in energy-distance/notebooks/train_sbert_coir_final.py

// Before running training, make sure the model, model_name, and hyperparameters (LR, scale) are correct.

sbatch energy-distance/notebooks/train_coir_final.sh



### D. Model Evaluation

// Modify paths and parameters in energy-distance/notebooks/eval_sbert_coir_final.py

sbatch energy-distance/notebooks/eval_final.sh





## IMPORTANT FILES
1. train_sbert_coir_final.py
2. train_coir_final.sh
3. eval_sbert_coir_final.py
4. eval_final.sh


## NOTES
scale parameter can be modified in
`sentence-transformers-3.4.1/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py`
