# Named entity recognition for machine learning interatomic potentials

This repo contains the codes for the paper 'Named entity recognition models for machine learning interatomic potentials: A user-
centric approach to knowledge extraction from scientific literature'.

## Requirements and Setup

- Python 3.10
- Pytorch (version 1.12.1)
- Transformers (version 4.41.2)

All required packages can be installed by running `conda env create -f environment.yml`.

## Running the code

The web application AnnoApp can be launched using the following command:
```bash
flask --app AnnoApp run
```

The script for the LLM fine-tuning can be run using the following command:

```bash
python finetune.py --save_model True --seed 111 --n_epochs 3 --learning_rate 0.00005
```


The MaterialsBERT language model used as the pre-trained model can be found [here](https://huggingface.co/pranav-s/MaterialsBERT).

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE.md) file for details.
