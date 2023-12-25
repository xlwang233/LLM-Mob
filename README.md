# ***L***arge ***L***anguage ***M***odels for Human ***Mob***ility Prediction (LLM-Mob)

Code for the paper ***[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)***.

The code is provided for reproducing the main results presented in the paper. However, the results may not be 100 per cent same as presented in the paper, due to the randomness of LLMs and the frequent update of OpenAI's GPT models. That being said, we anticipate that the difference is minimal.

## Data

The data is hosted in `/data`. As mentioned in our paper, we strictly follow the same data preprocessing steps in [Context-aware multi-head self-attentional neural network model for next location prediction](https://arxiv.org/abs/2212.01953). All the data files are generated from the data preprocessing scripts available [here](https://github.com/mie-lab/location-prediction). 

## Reproducing results on Geolife
### 1. Get an OpenAI account
If you already have an account and have set up API keys, skip this step. Otherwise, go to [OpenAI API website](https://openai.com/blog/openai-api) and sign up. Once you have an account, create an API key [here](https://platform.openai.com/account/api-keys). You may also need to set up your payment [here](https://platform.openai.com/account/billing/overview) in order to use the API. 
### 2. Run the scripts to start the prediction process.
Specify your OpenAI API Key in the beginning of the script `llm-mob.py`, change the parameters in the main function if necessary and start the prediction process by simply running the sripts
```bash
python llm-mob.py
```
The log file will be stored in `/logs` and prediction results will be stored in `/output`.

## Results and evaluation
We provide the actual prediction results obtained in our experiments in `/results`. 
To calculate the evaluation metrics, check the IPython notebook `metrics.ipynb` and run the scripts therein.

## Update on OpenAI API

OpenAI has recently released a new major version of their API, therefore the code in this repo has been updated accordingly.
For more information regarding how the update affect the old code and how we should proceed, check out their [v1.0.0 Migration Guide](https://github.com/openai/openai-python/discussions/742).

## Citation

```bibtex
@misc{wang2023i,
      title={Where Would I Go Next? Large Language Models as Human Mobility Predictors}, 
      author={Xinglei Wang and Meng Fang and Zichao Zeng and Tao Cheng},
      year={2023},
      eprint={2308.15197},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```