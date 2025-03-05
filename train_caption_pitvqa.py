import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import argparse
from Image_Captioning import create_dataset
import ruamel.yaml as yaml
from pathlib import Path
from Image_Captioning.models.blip import blip_decoder
import time
import datetime
import pandas as pd  
from captions_utils import * 

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(model, data_loader, optimizer, epoch, device):
    model.train()  
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train Caption Epoch: [{epoch}]'
    print_freq = 200

    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)       
        loss = model(image, caption)      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()         
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
  
    print("Averaged stats:", metric_logger.global_avg())     
    return metric_logger.meters['loss'].global_avg

@torch.no_grad()
def validate(model, data_loader, device, epoch, loss_history=None):
    model.eval() 
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Validate Epoch: [{epoch}]'
    print_freq = 50

    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        loss = model(image, caption)
        metric_logger.update(loss=loss.item())

    val_loss_avg = metric_logger.meters['loss'].global_avg
    print(f"Validation Epoch [{epoch}] - Loss: {val_loss_avg:.4f}")
    return val_loss_avg

def main(args, config):
    print(f"Use GPU:{args.gpu}") 
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print('Training PitVQA-Dataset')
    train_dataset, val_dataset, test_dataset = create_dataset('Image_Caption', format_style=args.format_style)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    print("Creating model")
    print(f"image_size: {config['image_size']}")
    print(f"image_encoder: {config['vit']}")
    print(f"prompt: {config['prompt']}")
    print(f'max_length: {args.max_length}')

    model = blip_decoder(
        pretrained=config['pretrained'],
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        prompt=config['prompt'],
        max_length=args.max_length
    ).to(device)

    model_without_ddp = model 
    optimizer = optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start training")
    start_time = time.time()  
    training_results = [] 

    for epoch in range(args.max_epoch):
        cosine_lr_schedule(optimizer, epoch, args.max_epoch, config['init_lr'], config['min_lr'])

        train_loss = train(model, train_loader, optimizer, epoch, device)
        val_loss = validate(model=model, data_loader=test_loader, device=device, epoch=epoch)

        training_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        checkpoint_path = f"{args.experiment_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(save_obj, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    results_df = pd.DataFrame(training_results)
    csv_path = os.path.join(args.experiment_dir, "training_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Training results saved to {csv_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_vqa_format.yaml')
    parser.add_argument('--output_dir', default='output/Caption_pitvqa')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use') 
    parser.add_argument('--experiment_name', default='default_experiment', help='Name of the experiment') 
    parser.add_argument('--format_style', default='refined_description', type=str, help='Specify the format style for the dataset') 
    parser.add_argument('--max_length', default=165, type=int, help="Max length token")
    parser.add_argument('--max_epoch', default=10, type=int, help="Max epoch for training")
    args = parser.parse_args()

    experiment_dir = os.path.join(args.output_dir, args.experiment_name) 
    args.experiment_dir = experiment_dir  
    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config)
