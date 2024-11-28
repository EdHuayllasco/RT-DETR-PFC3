python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --eval \
    -r rtdetr_r50vd_2x_coco_objects365.pdparams --use_vdl=True

#-r rtdetr_r18vd_5x_coco_objects365.pdparams #--use_vdl=True