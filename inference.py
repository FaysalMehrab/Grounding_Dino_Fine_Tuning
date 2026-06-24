import os
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

def get_text_dimensions(text, font, draw):
    """Helper to safely retrieve text dimensions across different Pillow versions."""
    try:
        # Pillow >= 10.0.0
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        return draw.textsize(text, font=font)

def draw_predictions(image, boxes, scores, labels):
    """Draws bounding boxes and high-contrast labels on the PIL image."""
    draw = ImageDraw.Draw(image)
    
    # Standardize a clean, scalable font (e.g., Arial) with solid fallbacks
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            # Explicit path fallback for Windows environments
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            
    # Professional color palette mapping (RGB format)
    color_palette = {
        "manniquin": (0, 255, 255),  # Neon Cyan
        "tent": (255, 165, 0),       # Vibrant Orange
        "default": (0, 255, 0)       # Bright Neon Green
    }
    
    for box, score, label in zip(boxes, scores, labels):
        box_coords = [round(x) for x in box.tolist()]
        score_val = score.item()
        class_name = label.lower().strip()
        
        # Get class color or use default neon green fallback
        color = color_palette.get(class_name, color_palette["default"])
        
        # 1. Draw thick bounding box border
        draw.rectangle(box_coords, outline=color, width=4)
        
        # 2. Draw solid background block for the label
        text = f" {label} {score_val:.2f} "
        text_w, text_h = get_text_dimensions(text, font, draw)
        
        # Position label banner slightly above the top edge of the bounding box
        # If the object is too close to the top edge, place the banner inside the box instead
        x1, y1 = box_coords[0], box_coords[1]
        banner_y1 = y1 - text_h - 4 if y1 - text_h - 4 > 0 else y1
        banner_x2 = x1 + text_w
        banner_y2 = banner_y1 + text_h + 4
        
        draw.rectangle([x1, banner_y1, banner_x2, banner_y2], fill=color)
        
        # 3. Draw high-contrast text on top of the banner
        draw.text((x1, banner_y1 + 2), text, fill=(0, 0, 0), font=font)

def run_batch_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dir = "sm_suas-1/test"
    output_dir = "Inference_Output"
    
    if not os.path.exists(test_dir):
        print(f"Error: test directory '{test_dir}' not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and sort target test image file list
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not test_images:
        print("Error: No images found in test directory.")
        return
        
    # Limit execution strictly to the first 20 test images
    target_images = test_images[:20]
    print(f"Loaded {len(target_images)} images for processing. Output will be saved in: '{output_dir}/'")
    
    # Load model and processor weights
    model_path = "./best_gdino_checkpoint"
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at '{model_path}'.")
        return
        
    processor = GroundingDinoProcessor.from_pretrained(model_path)
    model = GroundingDinoForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    text_prompt = "manniquin . tent ."
    box_threshold = 0.20
    text_threshold = 0.20
    
    # Process files sequentially with a progress tracking bar
    for image_name in tqdm(target_images, desc="Generating Inference Output"):
        image_path = os.path.join(test_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        target_sizes = [image.size[::-1]] # [height, width]
        
        # Prepare inputs and push to matching hardware device
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Version-safe post-processing fallback
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes
            )
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes
            )
            
        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result.get("text_labels", result.get("labels", []))
        
        # Apply visualization if detections exist
        if len(boxes) > 0:
            draw_predictions(image, boxes, scores, labels)
            
        # Save output maintaining the exact original filename
        output_path = os.path.join(output_dir, image_name)
        image.save(output_path)

    print(f"\nBatch processing complete. Output images are successfully stored in './{output_dir}/'")

if __name__ == "__main__":
    run_batch_inference()