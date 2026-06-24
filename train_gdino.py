import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

ImageFile.LOAD_TRUNCATED_IMAGES = True


# config
MAX_TRAIN_SAMPLES = 10 
MAX_VAL_SAMPLES = 2   
EPOCHS = 500
PATIENCE = 20
GRAD_ACCUMULATION_STEPS = 4  # Effective batch size of 4


class GroundingDinoCocoDataset(Dataset):
    def __init__(self, json_path, img_dir, processor, text_prompt, category_id_to_index, max_samples=None):
        self.img_dir = img_dir
        self.processor = processor
        self.text_prompt = text_prompt
        self.category_id_to_index = category_id_to_index
        
        with open(json_path, "r") as f:
            self.coco_data = json.load(f)
            
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        
        self.img_to_anns = {}
        for anno in self.coco_data["annotations"]:
            img_id = anno["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(anno)
            
        # Select images containing target classes
        self.img_ids = []
        for img_id in self.images.keys():
            if img_id in self.img_to_anns:
                valid_anns = [
                    a for a in self.img_to_anns[img_id] 
                    if a["category_id"] in self.category_id_to_index
                ]
                if len(valid_anns) > 0:
                    self.img_ids.append(img_id)
        
        if max_samples is not None:
            self.img_ids = self.img_ids[:max_samples]
        
    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_metadata = self.images[img_id]
        file_name = img_metadata["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        
        image = Image.open(img_path).convert("RGB")
        raw_anns = self.img_to_anns.get(img_id, [])
        
        formatted_anns = []
        for anno in raw_anns:
            cat_id = anno["category_id"]
            if cat_id not in self.category_id_to_index:
                continue
                
            bbox = anno["bbox"]
            mapped_cat_id = self.category_id_to_index[cat_id]
            area = anno.get("area", bbox[2] * bbox[3])
            
            formatted_anns.append({
                "category_id": mapped_cat_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": anno.get("iscrowd", 0),
                "id": anno.get("id", 0)
            })
            
        annotations = {
            "image_id": img_id,
            "annotations": formatted_anns
        }
        
        image_inputs = self.processor.image_processor(
            images=image,
            annotations=annotations,
            return_tensors="pt"
        )
        
        text_inputs = self.processor(
            text=self.text_prompt,
            return_tensors="pt"
        )
        
        item = {}
        item["pixel_values"] = image_inputs["pixel_values"][0]
        item["pixel_mask"] = image_inputs["pixel_mask"][0]
        item["labels"] = image_inputs["labels"][0]
        
        item["input_ids"] = text_inputs["input_ids"][0]
        item["attention_mask"] = text_inputs["attention_mask"][0]
        if "token_type_ids" in text_inputs:
            item["token_type_ids"] = text_inputs["token_type_ids"][0]
                
        return item

def collate_fn(batch):
    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "pixel_mask": torch.stack([item["pixel_mask"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": [item["labels"] for item in batch]
    }
    if "token_type_ids" in batch[0]:
        collated["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])
    return collated

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_dir = "sm_suas-1"
    train_json = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    train_img_dir = os.path.join(dataset_dir, "train")
    valid_json = os.path.join(dataset_dir, "valid", "_annotations.coco.json")
    valid_img_dir = os.path.join(dataset_dir, "valid")
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = GroundingDinoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id)
    
    target_classes = ["manniquin", "tent"]
    
    with open(train_json, "r") as f:
        train_coco_data = json.load(f)
        
    category_id_to_index = {}
    for cat in train_coco_data["categories"]:
        name = cat["name"].lower().strip()
        if name in target_classes:
            category_id_to_index[cat["id"]] = target_classes.index(name)
            
    text_prompt = " . ".join(target_classes) + " ."
    print(f"Target Prompt: '{text_prompt}'")
    
    train_dataset = GroundingDinoCocoDataset(
        train_json, train_img_dir, processor, text_prompt, category_id_to_index, max_samples=MAX_TRAIN_SAMPLES
    )
    valid_dataset = GroundingDinoCocoDataset(
        valid_json, valid_img_dir, processor, text_prompt, category_id_to_index, max_samples=MAX_VAL_SAMPLES
    )
    
    print(f"Dataset summary | Training images: {len(train_dataset)} | Validation images: {len(valid_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Freeze backbones
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        for param in model.model.backbone.parameters():
            param.requires_grad = False
            
    if hasattr(model, "model") and hasattr(model.model, "text_backbone"):
        for param in model.model.text_backbone.parameters():
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
    
    scaler = GradScaler()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} | Active Learning Rate: {current_lr:.2e} ---")
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for step, batch in enumerate(train_pbar):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "pixel_mask": batch["pixel_mask"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)
                
            inputs["labels"] = [
                {lk: lv.to(device) for lk, lv in label.items()}
                for label in batch["labels"]
            ]
            
            with autocast(device_type="cuda"):
                outputs = model(**inputs)
                loss = outputs.loss / GRAD_ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_train_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            train_pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in valid_pbar:
                inputs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "pixel_mask": batch["pixel_mask"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device)
                }
                if "token_type_ids" in batch:
                    inputs["token_type_ids"] = batch["token_type_ids"].to(device)
                    
                inputs["labels"] = [
                    {lk: lv.to(device) for lk, lv in label.items()}
                    for label in batch["labels"]
                ]
                
                with autocast(device_type="cuda"):
                    outputs = model(**inputs)
                    loss = outputs.loss
                    
                total_val_loss += loss.item()
                valid_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_val_loss = total_val_loss / len(valid_loader)
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Check validation improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_dir = "./best_gdino_checkpoint"
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Validation loss improved. Saved updated checkpoint to {checkpoint_dir}")
        else:
            epochs_no_improve += 1
            print(f"Early stopping status: {epochs_no_improve}/{PATIENCE} epochs without improvement")
            if epochs_no_improve >= PATIENCE:
                print(f"\nNo improvement in validation loss detected for {PATIENCE} epochs. Early stopping triggered.")
                break
                
    print("\nTraining execution complete.")

if __name__ == "__main__":
    train()