#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --test-only

python rtdetrv2_torch.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/best.pth \
        -f dataset/test/DirtyPanicleDisease-11-_jpg.rf.a18fc2ac8ed57dab6c39b0f8626f9baf.jpg \
        -d cuda