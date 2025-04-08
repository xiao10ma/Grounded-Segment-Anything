# 以下脚本将分别使用三种文本提示来处理图像
# 1. car
# 2. bus. truck
# 3. pedestrian

# 定义要使用的文本提示数组
TEXT_PROMPTS=("car" "bus. truck" "pedestrian")

# 循环处理每个文本提示
for text_prompt in "${TEXT_PROMPTS[@]}"; do
    echo "处理文本提示: $text_prompt"
    
    # GPU 0 处理相机0的图像
    CUDA_VISIBLE_DEVICES=5 python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --image_dir /HDD_DISK/users/mazipei/BZG/dataset/017/images \
        --output_dir  /HDD_DISK/users/mazipei/BZG/dataset/017/seg_npy \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$text_prompt" \
        --device "cuda" \
        --file_pattern "*_0.png" &

    # GPU 1 处理相机1的图像
    CUDA_VISIBLE_DEVICES=6 python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --image_dir /HDD_DISK/users/mazipei/BZG/dataset/017/images \
        --output_dir  /HDD_DISK/users/mazipei/BZG/dataset/017/seg_npy \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$text_prompt" \
        --device "cuda" \
        --file_pattern "*_1.png" &

    # GPU 2 处理相机2的图像
    CUDA_VISIBLE_DEVICES=7 python grounded_sam_demo.py \
        --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_checkpoint sam_vit_h_4b8939.pth \
        --image_dir /HDD_DISK/users/mazipei/BZG/dataset/017/images \
        --output_dir  /HDD_DISK/users/mazipei/BZG/dataset/017/seg_npy \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$text_prompt" \
        --device "cuda" \
        --file_pattern "*_2.png" &

    # 等待当前文本提示的所有后台进程完成
    wait
    
    echo "文本提示 '$text_prompt' 处理完成"
done

echo "所有处理已完成"
