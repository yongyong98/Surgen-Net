# SurGen-Net: A Generative Approach for Surgical VQA with Structured Text Generation

## Overview

SurGen-Net is a generative model designed for Surgical Visual Question Answering (VQA) with structured text generation. This model leverages advanced deep learning techniques to improve captioning and VQA performance in surgical contexts.

## Installation

Ensure you have the required dependencies installed before running the training or evaluation scripts.

```bash
pip install -r requirements.txt
```

## Training

Run the following command to train the model:

```bash
python train_caption_pitvqa.py --gpu 0 \
                               --config "./configs/caption_vqa_format.yaml" \
                               --experiment_name "20250305_mt_132_single_gpu_Pit_Caption_VQA_Format_epoch10_wrong" \
                               --format_style "refined_description_250225" \
                               --max_length 132 \
                               --max_epoch 10  
```

## Evaluation

To evaluate the model, use the following command:

```bash
python evaluate.py --config "./configs/caption_vqa_format.yaml" \
                   --experiment_name "20250305_mt_132_single_gpu_Pit_Caption_VQA_Format_epoch10_wrong" \
                   --device "cuda:0" \
                   --start_epoch 4 \
                   --end_epoch 10 \
                   --max_length 132 \
                   --format_style 'refined_description_250225'  
```

## Citation

If you use SurGen-Net in your research, please cite our work:

```
@article{your_paper,
  author    = {Your Name and Co-Authors},
  title     = {SurGen-Net: A Generative Approach for Surgical VQA with Structured Text Generation},
  journal   = {Your Journal},
  year      = {2025}
}
```

## Contact

For any questions or issues, please contact yjjj98\@gmail.com.

