python tools/infer.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    -o weights=output/rtdetr_r18vd_6x_arroz/best_model.pdparams \
    --infer_img=assets/healthy.jpg \
    --draw_threshold=0.05 \
    --visualize=True