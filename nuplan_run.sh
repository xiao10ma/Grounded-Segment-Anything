# text_prompt: [car, bus. truck, pedestrian]
# --text_prompt "car"
# --text_prompt "bus. truck"
# --text_prompt "pedestrian"

DATA_ROOT="/HDD_DISK/users/mazipei/BZG/data/nuplan/36b01" # b2730 36b01 c635b e1f65
OUTPUT_DIR="/HDD_DISK/users/mazipei/BZG/data/nuplan/36b01/seg_npy"

# dir: image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7

GPU_LIST=(0 1 2 3 4 5 6 7)

for i in {0..7}; do
    mkdir -p "$OUTPUT_DIR/mask_$i"

    # 将命令放到后台执行
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --image_dir "$DATA_ROOT/image_$i" \
        --output_dir "$OUTPUT_DIR/mask_$i" \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "car" \
        --device "cuda" &
done

# 等待所有后台进程执行结束
wait

echo "All tasks have finished!"
