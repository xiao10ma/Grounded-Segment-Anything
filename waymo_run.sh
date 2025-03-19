# text_prompt: [car, bus. truck, pedestrian]
# --text_prompt "car"
# --text_prompt "bus. truck"
# --text_prompt "pedestrian"

# GPU 0 处理相机0的图像
CUDA_VISIBLE_DEVICES=0 python grounded_sam_demo.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --image_dir /HDD_DISK/users/mazipei/BZG/dataset/703/images \
    --output_dir  /HDD_DISK/users/mazipei/BZG/dataset/703/seg_npy \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --text_prompt "car" \
    --device "cuda" \
    --file_pattern "*_0.png" &

# GPU 1 处理相机1的图像
CUDA_VISIBLE_DEVICES=1 python grounded_sam_demo.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --image_dir /HDD_DISK/users/mazipei/BZG/dataset/703/images \
    --output_dir  /HDD_DISK/users/mazipei/BZG/dataset/703/seg_npy \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --text_prompt "car" \
    --device "cuda" \
    --file_pattern "*_1.png" &

# GPU 2 处理相机2的图像
CUDA_VISIBLE_DEVICES=2 python grounded_sam_demo.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --image_dir /HDD_DISK/users/mazipei/BZG/dataset/703/images \
    --output_dir  /HDD_DISK/users/mazipei/BZG/dataset/703/seg_npy \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --text_prompt "car" \
    --device "cuda" \
    --file_pattern "*_2.png" &

# 等待所有后台进程完成
wait
