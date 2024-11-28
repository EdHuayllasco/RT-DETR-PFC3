python tools/export_model.py \
            -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
            -o weights=weights/rtdetr_r18vd_5x_coco_objects365.pdparams \
            trt=True \
            --output_dir=output_inference

python scripts/paddle_infer_shape.py \
        --model_dir=./output_inference/rtdetr_r18vd_6x_coco \
        --model_filename model.pdmodel  \
        --params_filename model.pdiparams \
        --save_dir rtdetr_r18_gun \
        --input_shape_dict="{'image':[-1,3,640,640]}"