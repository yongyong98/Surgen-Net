import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
   
class PitVQA_Image_Caption_VQA_Dataset(Dataset):
    def __init__(self, csv_path, image_root, format_style, transform=None):
        """
        Args:
            csv_path (str): CSV 파일 경로
            image_root (str): 이미지가 저장된 상위 디렉토리
            transform (callable, optional): 이미지에 적용할 변환 함수
        """
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.format_style = format_style
        print(f'format_style: {format_style}')

    def __len__(self):
        """데이터셋의 길이 반환"""
        return len(self.data)

    def __getitem__(self, index):
        """
        주어진 index의 이미지와 캡션 반환
        Args:
            index (int): 데이터셋의 인덱스
        Returns:
            image (PIL Image or Tensor): 변환된 이미지
            caption (str): 이미지에 대한 캡션
        """
        row = self.data.iloc[index]
        
        image_path = os.path.join(self.image_root, row['video'], row['image'])
        image_path = os.path.splitext(image_path)[0] + '.png'
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = row[self.format_style] 
        return image, caption
    
class PitVQA_Image_Caption_Dataset_eval(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        """
        Args:
            csv_path (str): CSV 파일 경로
            image_root (str): 이미지가 저장된 상위 디렉토리
            transform (callable, optional): 이미지에 적용할 변환 함수
        """
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        """데이터셋의 길이 반환"""
        return len(self.data)

    def __getitem__(self, index):
        """
        주어진 index의 이미지와 캡션 반환
        Args:
            index (int): 데이터셋의 인덱스
        Returns:
            image (PIL Image or Tensor): 변환된 이미지
            caption (str): 이미지에 대한 캡션
        """
        row = self.data.iloc[index]
        
        image_path = os.path.join(self.image_root, row['video'], row['image'])
        image_path = os.path.splitext(image_path)[0] + '.png'

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        img_id = row['video'] + '_' + row['image']
        return image, img_id
    
## PitVQA_Grad_Cam
class Grad_Cam_PitVQA_Image_Caption_VQA_Dataset(Dataset):
    def __init__(self, csv_path, image_root, format_style, transform=None):
        """
        Args:
            csv_path (str): CSV 파일 경로
            image_root (str): 이미지가 저장된 상위 디렉토리
            transform (callable, optional): 이미지에 적용할 변환 함수
        """
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.format_style = format_style
        print(f'format_style: {format_style}')

    def __len__(self):
        """데이터셋의 길이 반환"""
        return len(self.data)

    def __getitem__(self, index):
        """
        주어진 index의 이미지와 캡션 반환
        Args:
            index (int): 데이터셋의 인덱스
        Returns:
            image (PIL Image or Tensor): 변환된 이미지
            caption (str): 이미지에 대한 캡션
        """
        row = self.data.iloc[index]
        
        image_path = os.path.join(self.image_root, row['video'], row['image'])
        image_path = os.path.splitext(image_path)[0] + '.png'
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = row[self.format_style] 
        return image, caption, image_path
