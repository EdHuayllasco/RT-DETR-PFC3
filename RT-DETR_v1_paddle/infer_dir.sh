python tools/infer.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    -o weights=output/rtdetr_r18vd_6x_arroz/best_model.pdparams \
    --infer_dir=dataset/arroz_class/test/blast \
    --output_dir=output/blast \
    --draw_threshold=0.30 \
    --save_results true