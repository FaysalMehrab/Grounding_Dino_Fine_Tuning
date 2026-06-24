import os
import json
from collections import Counter

def analyze_split(split_name, json_path, img_dir):
    print(f"\n================ Analyzing Split: {split_name} ================")
    if not os.path.exists(json_path):
        print(f"Error: Annotation file not found at {json_path}")
        return
    
    with open(json_path, "r") as f:
        coco_data = json.load(f)
    
    # 1. Categories
    print("\n[Categories in JSON]")
    categories = coco_data.get("categories", [])
    cat_id_to_name = {}
    for cat in categories:
        print(f"  - Category ID: {cat['id']}, Name: '{cat['name']}', Supercategory: '{cat.get('supercategory', 'N/A')}'")
        cat_id_to_name[cat["id"]] = cat["name"]
        
    # 2. Metadata Counts
    num_images = len(coco_data.get("images", []))
    num_annotations = len(coco_data.get("annotations", []))
    print(f"\n[Metadata Counts]")
    print(f"  - Total Images listed: {num_images}")
    print(f"  - Total Annotations: {num_annotations}")
    
    # Check physical image files
    if os.path.exists(img_dir):
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  - Physical image files in directory: {len(files)}")
    else:
        print(f"  - Image directory '{img_dir}' does not exist.")
        
    # 3. Class Distribution
    print("\n[Annotation Distribution per Class]")
    class_counts = Counter()
    for ann in coco_data.get("annotations", []):
        cat_id = ann["category_id"]
        class_name = cat_id_to_name.get(cat_id, f"Unknown (ID: {cat_id})")
        class_counts[class_name] += 1
        
    for name, count in class_counts.items():
        print(f"  - Class '{name}': {count} bounding boxes")
        
    # 4. Bounding Box Statistics
    if num_annotations > 0:
        widths, heights = [], []
        for ann in coco_data.get("annotations", []):
            bbox = ann["bbox"] # COCO: [x, y, w, h]
            widths.append(bbox[2])
            heights.append(bbox[3])
        print("\n[Bounding Box Statistics (Pixels)]")
        print(f"  - Min width: {min(widths):.1f}, Max width: {max(widths):.1f}, Avg width: {sum(widths)/len(widths):.1f}")
        print(f"  - Min height: {min(heights):.1f}, Max height: {max(heights):.1f}, Avg height: {sum(heights)/len(heights):.1f}")

def show_directory_structure(base_dir):
    print(f"\n================ Directory Structure of '{base_dir}' ================")
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        sub_dir = os.path.basename(root) if os.path.basename(root) else base_dir
        print(f"{indent}└── {sub_dir}/")
        sub_indent = ' ' * 4 * (level + 1)
        # Just show count of image files to avoid terminal flooding
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        jsons = [f for f in files if f.endswith('.json')]
        if jsons:
            for j in jsons:
                print(f"{sub_indent}├── {j}")
        if images:
            print(f"{sub_indent}├── [{len(images)} images]")

if __name__ == "__main__":
    dataset_dir = "sm_suas-1"
    show_directory_structure(dataset_dir)
    
    analyze_split("Train", os.path.join(dataset_dir, "train", "_annotations.coco.json"), os.path.join(dataset_dir, "train"))
    analyze_split("Validation", os.path.join(dataset_dir, "valid", "_annotations.coco.json"), os.path.join(dataset_dir, "valid"))