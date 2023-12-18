# Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks

<h1 align="center"><img src="../docs/images/intro.png" width="75%"></h1>

[[Paper](https://openreview.net/pdf?id=ONwL9ucoYG)]

## Overview
ROCLIP is the **first** effective method for
robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks.

### Training 

```
python -m src.main --name exp1 --train_data <path to train csv file> --validation_data <path to valid csv file>
--image_key <column name of the image paths in the train/validation csv file> --caption_key <column name of the captions
in the train/validation csv file> --device_ids 0 1 2 3 --distributed --memory_bank_size <memory bank size> --break_epoch <RoCLIP frequency> --memory_bank  --RoCLIP_switch --cross_aug   
```

Your train/validation csv/tsv file should have 2 columns containing captions and the path to corresponding images on the machine. this script does not download the images for the captions directly. To download the images from their URL for CC3M and/or CC12M, use our `utils/download.py` script.

### Inference - ImageNet1K

```
python -m src.main --name <eval_imagenet_1k> --eval_data_type <dataset> --eval_test_data_dir data/ImageNet1K/validation/ --device_id 0 --checkpoint <ckpts/epoch_64.pt> 
```

For ImageNet1K: There should be a labels.csv in the test data directory that contains 2 columns -- image, label. image should have the location to the image in the local machine.

## Acknowledgements
Some code in this repo comes from the following repositories:

[CyCLIP](https://github.com/goel-shashank/CyCLIP)

[mlfoundations](https://github.com/mlfoundations/open_clip)  

[openai](https://github.com/openai/CLIP)

## Setup Environment and Install dependencies

The following commands create a conda environment inside the repository with the dependencies.

```bash
conda env create --prefix ./env -f environment.yml
source activate ./env
```

Then, to download the missing nltk_data

```
python -m nltk.downloader all

```

# Citation

Please cite our paper if you find the results or our code useful.
```

@inproceedings{yang2023robust,
  title={Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks},
  author={Yang, Wenhan and Gao, Jingdong and Mirzasoleiman, Baharan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```