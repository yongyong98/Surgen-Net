import os
import torch
from torch.utils.data import DataLoader
from Image_Captioning import *
from Image_Captioning.models.blip import blip_decoder
import ruamel.yaml as yaml
import json
from tqdm import tqdm
import argparse


def save_result(result, result_dir, filename, remove_duplicate):
    """
    Saves the result as a JSON file.
    """
    result_file = os.path.join(result_dir, f'{filename}.json')
    json.dump(result, open(result_file, 'w'))
    return result_file


@torch.no_grad()
def evaluate(model, data_loader, device, config, args):
    """
    Evaluates the model and returns the generated captions.
    """
    model.eval()
    result = []
    print(f'max_length: {args.max_length}')
    for image, image_id in tqdm(data_loader, desc="Processing images"):
        image = image.to(device)
        captions = model.generate(
            image, sample=False, num_beams=config['num_beams'],
            max_length=args.max_length, min_length=config['min_length']
        )
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id, "caption": caption})
    return result


def load_checkpoint(checkpoint_path, model):
    """
    Loads a checkpoint and restores the model state.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Checkpoint loaded successfully!")
    return checkpoint['epoch']


def evaluate_from_checkpoint(config_path, checkpoint_path, result_dir, args, device='cuda:0'):
    """
    Loads a checkpoint, initializes the model, and performs evaluation.
    """
    device = torch.device(device)

    print("Loading configuration...")
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    print("Creating PitVQA captioning dataset...")
    _, _, test_dataset = create_dataset('Image_Caption', format_style=args.format_style)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
    )

    print("Initializing model...")
    model = blip_decoder(
        pretrained=config['pretrained'],
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        prompt=config['prompt']
    ).to(device)

    start_epoch = load_checkpoint(checkpoint_path, model)
    print("Starting evaluation...")
    test_result = evaluate(model, test_loader, device, config, args)

    os.makedirs(result_dir, exist_ok=True)
    result_file = save_result(test_result, result_dir, f'test_result_epoch{start_epoch}', remove_duplicate='image_id')
    print(f"Test results saved at: {result_file}")

    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint for image captioning.")
    parser.add_argument('--config', default="configs/caption_vqa_format.yaml", help="Path to the configuration YAML file.")
    parser.add_argument('--root_dir', default="C:/Users/kyuhw/Desktop/work/PitVQA-main/output/Caption_pitvqa/", help="Root directory for the experiment outputs.")
    parser.add_argument('--experiment_name', required=True, help="Experiment name to identify the outputs.")
    parser.add_argument('--device', default='cuda:0', help="Device to run the evaluation (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--start_epoch', type=int, default=0, help="Start epoch for evaluation.") 
    parser.add_argument('--end_epoch', type=int, default=10, help="End epoch for evaluation.")  
    parser.add_argument('--max_length', default=165, type=int, help="Max length token")
    parser.add_argument('--format_style', default='refined_description', type=str, help='Specify the format style for the dataset')  
    args = parser.parse_args()

    config_path = args.config
    root_dir = args.root_dir
    experiment_name = args.experiment_name
    result_dir = os.path.join(root_dir, experiment_name)

    for epoch in range(args.start_epoch, args.end_epoch):
        checkpoint_path = os.path.join(root_dir, experiment_name, f"checkpoint_epoch_{epoch}.pth")
        evaluate_from_checkpoint(config_path, checkpoint_path, result_dir, args=args, device=args.device)  
