

# [WIP] STAMP Your Content: Proving Dataset Membership via Watermarked Rephrasings

The repo contains the official code for the ICML 25 paper [STAMP Your Content: Proving Dataset Membership via Watermarked Rephrasings](https://arxiv.org/abs/2504.13416) by Saksham Rastogi, Pratyush Maini, and Danish Pruthi.

### Setup

To install the necessary packages, first create a conda environment.
```
conda create -n <env_name> python=3.10
conda activate <env_name>
```
Then, install the required packages with 
```
pip install -r requirements.txt
```

## Artifacts
We provide the following artifacts for future research and reproducibility:

### Models

Below are the links to trained models (continual pretraining on contaminated data) from the paper's experiments (hosted on huggingface). They can also be found at this [Hugging Face Collection](https://huggingface.co/collections/p1xelsr/stamp-your-content-683e7c0a95d42e0411276813).

#### Pythia 1B models contaminated with benchmarks

- [Continual pretraining on corpus of ~6B tokens.](https://huggingface.co/p1xelsr/wtm_gamma0.25_delta1.0_6m)
- [Continual pretraining on corpus of ~4B tokens.](https://huggingface.co/p1xelsr/wtm_gamma0.25_delta1.0_4m)
- [Continual pretraining on corpus of ~3B tokens.](https://huggingface.co/p1xelsr/wtm_gamma0.25_delta1.0_3m)
- [Continual pretraining on corpus of ~2B tokens.](https://huggingface.co/p1xelsr/wtm_gamma0.25_delta1.0_2m)
- [Continual pretraining on corpus of ~1B tokens.](https://huggingface.co/p1xelsr/wtm_gamma0.25_delta1.0_1m)

### Datasets

- The `benchmarks` folder contains all the test files used to produce the paper's results,
including both original and rephrased versions for the following four datasets:

  - [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)
  - [GSM8K](openai/gsm8k)
  - [MMLU](cais/mmlu)
  - [TriviaQA](mandarjoshi/trivia_qa)

## Acknowledgements

We heavily rely on the following repos in our paper:
1. [LLM Dataset Inference](https://github.com/pratyushmaini/llm_dataset_inference)
2. [MarkLLM](https://github.com/THU-BPM/MarkLLM)

## Issues

If you have any questions, feel free to open an issue on GitHub or contact Saksham  (iitdsaksham@gmail.com).

## Reference

If you find this repo useful, please consider citing:

```bibtex
@misc{rastogi2025stampcontentprovingdataset,
      title={STAMP Your Content: Proving Dataset Membership via Watermarked Rephrasings}, 
      author={Saksham Rastogi and Pratyush Maini and Danish Pruthi},
      year={2025},
      eprint={2504.13416},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.13416}, 
}
```