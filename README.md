# Estimating intracranial pressure using OCT scans of the eyeball

Members: Eric He, Farris Atif, Nasser Al-Rayes, Zixiao Chen

Estimate intracranial pressure (ICP) given OCT scans and intraocular pressure (IOP) values. See our [project poster](docs/poster.pdf).

## Repository Organization
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Presentation information
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks for cleaning data and exploratory data analysis
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── data           <- Scripts to download or generate data
│   ├── models         <- Scripts to train models and then use trained models to make
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Important links
[Google Drive](https://drive.google.com/drive/folders/1Z4KUKsvsiuYkY5aSK6cYcpvNEO4hjrWa?usp=sharing): holds raw data, reports
- [Raw data](https://drive.google.com/drive/folders/1NbXpNWhL59gayG6hd93Fx8Qhe6neTq76?usp=sharing): copy of the original monkey scans
- [PyTorch reduction](https://drive.google.com/drive/folders/1TZ2Np_obz5Sav3WzhgZjs787UGmNcq_B?usp=sharing): PyTorch arrays of reduced monkey scans, NOT STANDARDIZED
- [PyTorch standardized images](https://drive.google.com/drive/folders/1-6uMek90sNsCLCWNBisJbvna7hUndieS?usp=sharing): Pytorch tensors of reduced and standardized monkey scans (what we use for training)
- [Image samples](https://drive.google.com/drive/folders/11ZMbQv25VAaZhsd5WMYpXtI-KWhA2jzU?usp=sharing): Examples of images after downsizing (but not standardizing) - taken from the PyTorch reduction folder
- [Master Dataset](https://docs.google.com/spreadsheets/d/1PJHEbsb_w-g312iIb2SMkWq17OO_iZl-mim8NyuDAzI/edit?usp=sharing): holds master mappings from the raw data to our image samples, with IOP values filled in. Replicated to [the repository](data/monkey_data.csv)

## Data processing
[PyTorch reduction](src/notebooks/2021_11_20_eric_crop_iages.ipynb): code to downsize OCTs into PyTorch reduced arrays

## Model training
[From-scratch training](train.py): runs training using our [forked 3-D resnet training code](src/models/from_scratch/resnet_for_multimodal_regression.py)

[Pre-trained](src/models/MedicalNet/train.py): runs training using our [forked MedicalNet model](src/models/MedicalNet/models/resnet.py)

[Self-supervised](src/moco/): use the MoCo self-supervised learning to pre-train model