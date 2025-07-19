import argparse
import os
import sys
import concurrent.futures
import threading
import glob

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加全局线程锁
gpu_lock = threading.Lock()

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data_merge(output_dir, mask_list, box_list, label_list):
    # Function for merging masks with existing ones
    # Load original mask
    ori_mask_path = os.path.splitext(output_dir)[0] + '.npy'
    ori_mask = np.load(ori_mask_path)
    value = ori_mask.max()
    
    for idx, mask in enumerate(mask_list):
        # Convert mask to numpy array
        mask_np = mask.cpu().numpy()[0]
        
        # Check if mask area already has values in original mask
        overlap_region = ori_mask[mask_np]
        non_zero_pixels = np.sum(overlap_region > 0)
        
        # Calculate total mask pixels
        mask_pixels = np.sum(mask_np)
        
        # Skip if >90% overlap with existing mask
        if non_zero_pixels > 0 and non_zero_pixels > 0.9 * mask_pixels:
            continue
            
        # Update original mask
        ori_mask[mask_np] = value + idx + 1
    
    # Save numeric mask
    mask_save_path = os.path.splitext(output_dir)[0] + '.npy'
    np.save(mask_save_path, ori_mask)
    
    # Save visualization image
    plt.figure(figsize=(10, 10))
    plt.imshow(ori_mask)
    plt.axis('off')
    plt.savefig(output_dir, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

def save_mask_data_overwrite(output_dir, mask_list, box_list, label_list):
    # Function for overwriting masks
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:], dtype=torch.uint8)

    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    
    # Save numeric mask
    mask_save_path = os.path.splitext(output_dir)[0] + '.npy'
    np.save(mask_save_path, mask_img)
    
    # Save visualization image
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img)
    plt.axis('off')
    plt.savefig(output_dir, bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

def save_mask_data(output_dir, mask_list, box_list, label_list):
    # Auto-select function based on file existence
    mask_save_path = os.path.splitext(output_dir)[0] + '.npy'
    
    if os.path.exists(mask_save_path):
        # Use merge mode if file exists
        save_mask_data_merge(output_dir, mask_list, box_list, label_list)
    else:
        # Use overwrite mode if file doesn't exist
        save_mask_data_overwrite(output_dir, mask_list, box_list, label_list)

def process_single_image(args, model, predictor, image_path):
    try:
        img = os.path.basename(image_path)
        if image_path[-4] == '3' or image_path[-4] == '4':
            return
        print(f"Processing {image_path}")
        # load image
        image_pil, image = load_image(image_path)

        # 使用线程锁保护GPU操作
        with gpu_lock:
            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, args.text_prompt, args.box_threshold, args.text_threshold, device=args.device
            )

            # If no objects are detected, save a black image and continue to the next image
            if len(boxes_filt) == 0:
                print(f"No objects detected in {img}")
                h, w = image_pil.size[1], image_pil.size[0]
                print(h, w)
                mask_img = torch.zeros((1, 1, h, w), dtype=torch.bool)  # init to black image
                output_path = os.path.join(args.output_dir, img)
                save_mask_data(output_path, mask_img, boxes_filt, pred_phrases)
                return

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(args.device)

            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(args.device),
                multimask_output = False,
            )
            
            output_path = os.path.join(args.output_dir, img)
            save_mask_data(output_path, masks, boxes_filt, pred_phrases)
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=False, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image directory")
    parser.add_argument("--file_pattern", type=str, default="*", help="File name pattern, e.g. '*_0.png'")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_dir = args.image_dir
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    # 修改图像文件列表筛选逻辑，使用glob模块进行模式匹配
    pattern_path = os.path.join(image_dir, args.file_pattern)
    image_files = glob.glob(pattern_path)
    
    print(f"找到{len(image_files)}个匹配的图像文件: {args.file_pattern}")

    # Use ThreadPoolExecutor for parallel processing
    # 减少并发线程数量，避免资源竞争
    max_workers = min(4, os.cpu_count(), len(image_files))  # 限制最大线程数为4
    print(f"Processing {len(image_files)} images with {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_image, args, model, predictor, image_path)
            for image_path in image_files
        ]
        # 等待所有任务完成
        concurrent.futures.wait(futures)

    print("All images processed successfully!")

