CUDA_VISIBLE_DEVICES=0 torchrun --master-port=8989 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -t rtdetrv2_r18vd_120e_coco_rerun_48.1.pth -d cuda --output-dir ./output
