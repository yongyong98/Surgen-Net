'''
Description: Dataloader of PitVQA-Net model
Paper: PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery
Author: Runlong He, Mengya Xu, Adrito Das, Danyal Z. Khan, Sophia Bano, 
        Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam
Lab: Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS), UCL
Acknowledgement : Code adopted from the official implementation of 
                  Huggingface Transformers (https://github.com/huggingface/transformers)
                  and Surgical-GPT (https://github.com/lalithjets/SurgicalGPT).
'''

import os
import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
torchvision.disable_beta_transforms_warning()

from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
import torch 

class Pit24VQAClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail):

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + curr_seq + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                answer = line.split('|')[1]
                if answer not in ['no_visible_instrument', 'no_secondary_instrument']:  # filter unknown answers
                    self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

        # Labels
        self.labels = [
            'nasal_corridor_creation', 'anterior_sphenoidotomy', 'septum_displacement', 'sphenoid_sinus_clearance',
            'sellotomy', 'haemostasis', 'synthetic_graft_placement', 'durotomy', 'tumour_excision',
            'fat_graft_placement', 'gasket_seal_construct', 'dural_sealant', 'nasal_packing', 'debris_clearance',
            'end_of_step',  # 15 steps
            'nasal_sphenoid', 'sellar', 'closure',  'end_of_phase',  # 4 phases
            'suction', 'freer_elevator', 'pituitary_rongeurs', 'spatula_dissector', 'kerrisons', 'cottle',
            'haemostatic_foam', 'micro_doppler', 'nasal_cutting_forceps', 'stealth_pointer', 'irrigation_syringe',
            'retractable_knife', 'dural_scissors', 'ring_curette', 'cup_forceps', 'bipolar_forceps', 'tissue_glue',
            'surgical_drill',  # 18 instruments
            'zero', 'one', 'two',  # 3 number of instruments
            'top-left', 'top-right', 'centre', 'bottom-left', 'bottom-right',  # 5 positions
            'The middle and superior turbinates are laterally displaced',
            'The sphenoid ostium is identified and opened', 'The septum is displaced until the opposite ostium is seen',
            'The sphenoid sinus is opened, with removal of sphenoid septations to expose the face of the sella and mucosa',
            'Haemostasis is achieved with a surgiflo, a bipolar cautery, and a spongostan placement',
            'The sella is identified, confirmed and carefully opened', 'A cruciate durotomy is performed',
            'The tumour is seen and removed in a piecemeal fashion', 'spongostan, tachosil and duragen placement',
            'A fat graft is placed over the defact', 'Evicel and Adherus dural sealant are applied',
            'Debris is cleared from the nasal cavity and choana', 'A MedPor implant and a fascia lata graft are placed',
            'The nasal cavity is packed with Bismuth soaked ribbon gauze'  # 14 operations
        ]

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        video_num = qa_full_path.parts[-2]
        file_name = self.vqas[idx][0].split('/')[-1]

        img_loc = os.path.join(seq_path, 'images', file_name.split('.')[0] + '.png')
        img_loc = img_loc.replace('\\', '/')

        if not os.path.exists(img_loc):
            raise FileNotFoundError(f"Image file not found: {img_loc}")
        
        raw_image = Image.open(img_loc).convert('RGB')
        img = self.transform(raw_image)

        question = self.vqas[idx][1].split('|')[0]
        answer = self.vqas[idx][1].split('|')[1]
        label = self.labels.index(str(answer))

        return img_loc, img, question, label