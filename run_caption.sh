python train_caption_pitvqa.py --gpu 0 \
                               --config "./configs/caption_vqa_format.yaml" \
                               --experiment_name "20250305_mt_132_single_gpu_Pit_Caption_VQA_Format_epoch10_wrong" \
                               --format_style "refined_description_250225" \
                               --max_length 132 \
                               --max_epoch 10 

python evaluate.py --config "./configs/caption_vqa_format.yaml" \
                   --experiment_name "20250305_mt_132_single_gpu_Pit_Caption_VQA_Format_epoch10_wrong" \
                   --device "cuda:0" \
                   --start_epoch 4 \
                   --end_epoch 10 \
                   --max_length 132 \
                   --format_style 'refined_description_250225'