# RobustUIE

This is the official code release of the following paper:

Towards Robust Universal Information Extraction: Dataset, Evaluation, and Solution

## Usage

### Environment Setup

```
conda create -n RobustUIE python=3.10

activate RobustUIE

pip install -r requirements.txt
```

### Dataset Preparation

#### 1. Training Data

Note: Because some datasets have copyright requirements and need licenses, we cannot directly release this part of the data now. If you have a license for restricted datasets, you can use them to contact emails in Contact to obtain data.

We conduct experiments utilizing 7 datasets, comprising 3 datasets for the NER task, 2 datasets for the RE task, and 2 datasets for the ED tasks. 

#### 2. RUIE-Bench

We evaluate the robustness of the IE models using our constructed RUIE-Bench. You can download the data from [Google Drive](https://drive.google.com/file/d/1l8oUDkhXjZkW4fnQ2X0kd-MryTEWR-_t).

### Retrieval File Preparation

1. Please download the retrieval-related models from `https://huggingface.co/sentence-transformers/all-mpnet-base-v2` and `https://huggingface.co/flair/ner-english-ontonotes-large`, and put them to the folder `retrieval_models`.
2. Please install the retrieval-related package: `conda install -c pytorch faiss-gpu` (Linux) or `conda install -c pytorch faiss-cpu` (Windows).
3. Then, run `python get_dataset_embed.py` to generate retrieval files. Note that variables `dataset_name_list` and `datapath` in lines 16 and 18 should be changed according to the task and datasets.

### Prompt Construction

Run `python [task]_Prompt.py` to construct prompts for each task accordingly.

Necessary arguments are:

* `dataset`: Name of the dataset.
* `test_file`: File path of the test set.
* `train_file`: File path of the train set.
* `retrieval_strategy`: Retrieval strategy, including `'random'`, `'sentence_emb'`, and `'anonymized_sentence_emb'`.
* `output_file`: The output file path.
* `incontext_examples_num`: The number of in-context examples.

### Information Extraction

Run `python get_extraction_result.py` to interact with LLMs and get extraction results. Note that the value of variable `openai.api_key` in line 6 should be filled.

Necessary arguments are:

* `model`: Name of the dataset.
* `task`: Name of the task, including `'NER'`, `'RE'`, `'EAE'` and `'EE'`.
* `dataset`: Name of the dataset.
* `prompt_type`: Type of the prompt, including `'1stage'`, `'2stage'`, and `'1&2stage'`.
* `input_file`: The input file path.
* `output_file`: The output file path.
