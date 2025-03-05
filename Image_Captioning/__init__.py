import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .randaugment import RandomAugment
from .Image_Caption_dataset import * 

def create_dataset(dataset, format_style=None):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711))
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 
                                              'Sharpness', 'Equalize', 'ShearX', 
                                              'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])  
        
    if dataset == 'Image_Caption':
        image_root = r"C:\Users\kyuhw\Desktop\work\PitVQA-Dataset\images"
        train_csv_path = r"C:\Users\kyuhw\Desktop\work\PitVQA-Dataset\image_caption\train_pit_qa_revision_refined_data_250225_wrong.csv"
        test_csv_path = r"C:\Users\kyuhw\Desktop\work\PitVQA-Dataset\image_caption\test_pit_qa_revision_refined_data_250225_wrong.csv"

        train_dataset = PitVQA_Image_Caption_VQA_Dataset(train_csv_path, image_root, format_style, transform=transform_train)
        val_dataset = PitVQA_Image_Caption_VQA_Dataset(test_csv_path, image_root, format_style, transform=transform_test)
        test_dataset = PitVQA_Image_Caption_Dataset_eval(test_csv_path, image_root, transform=transform_test)  
    
        return train_dataset, val_dataset, test_dataset 
    
    elif dataset == 'Grad_CAM_PIT':
        image_root = r"C:\Users\kyuhw\Desktop\work\PitVQA-Dataset\images"
        train_csv_path = r"C:\Users\kyuhw\Desktop\work\PitVQA-Dataset\image_caption\train_pit_qa_revision_refined_data_all.csv"
        test_csv_path = r"C:\Users\kyuhw\Desktop\work\PitVQA-Dataset\image_caption\test_pit_qa_revision_refined_data_all.csv"

        train_dataset = Grad_Cam_PitVQA_Image_Caption_VQA_Dataset(train_csv_path, image_root, format_style, transform=transform_train)
        val_dataset = Grad_Cam_PitVQA_Image_Caption_VQA_Dataset(test_csv_path, image_root, format_style, transform=transform_test)
        test_dataset = PitVQA_Image_Caption_Dataset_eval(test_csv_path, image_root, transform=transform_test)
    
        return train_dataset, val_dataset, test_dataset 

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers
