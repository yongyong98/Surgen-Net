import os
import json
import re
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score

# Function to convert CSV file to COCO-style JSON format for ground truth
def csv_to_json(csv_file_path, output_json_path, format_style='refined_description'):
    """
    Convert CSV file to COCO-style JSON format.

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_json_path (str): Path to the output JSON file.
        format_style (str): Column name in CSV to use for caption.
    """
    df = pd.read_csv(csv_file_path)
    json_format = []
    
    # Generate image_id and extract caption
    for _, row in df.iterrows():
        image_id = f"{row['video']}_{row['image']}"
        caption = row[format_style]
        json_format.append({"image_id": image_id, "caption": caption})
    
    # Save the COCO-style JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_format, f, indent=4)
    
    print(f"COCO format JSON saved at '{output_json_path}'")

# Function to parse a caption into a dictionary of key-value pairs
def parse_caption(caption: str) -> dict:
    """
    Parse a caption into key-value pairs.

    Args:
        caption (str): The caption string to parse.

    Returns:
        dict: Dictionary with keys as question labels and values as normalized answers.
    """
    keys = [
        "what is the current phase?", "what is the current step?", "what is instrument1?",
        "where is instrument1?", "what is instrument2?", "where is instrument2?",
        "what is the current action?", "what is the next phase?", "what is the next step?",
        "the number of instrument"
    ]
    
    parsed = {k: None for k in keys}
    for line in caption.split('\n'):
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            value = normalize_value(value)
            for k in keys:
                if re.sub(r'[\s_]+', '', key.lower()) == re.sub(r'[\s_]+', '', k.lower()):
                    parsed[k] = value
                    break
    return parsed

# Function to normalize text by converting to lowercase and removing extra spaces
def normalize_value(val: str) -> str:
    """
    Normalize text by converting to lowercase and removing extra spaces.

    Args:
        val (str): The text value to normalize.

    Returns:
        str: Normalized text.
    """
    val = val.lower().strip()
    val = re.sub(r'\s*_\s*', '_', val)
    val = re.sub(r'\s*-\s*', '-', val)
    val = re.sub(r'\s+', ' ', val)
    return val

# Function to compute evaluation metrics
def compute_metrics(gt_dict, pred_dict, keys):
    """
    Compute evaluation metrics for predictions against ground truth.

    Args:
        gt_dict (dict): Dictionary of ground truth captions parsed.
        pred_dict (dict): Dictionary of predicted captions parsed.
        keys (list): List of keys to evaluate.

    Returns:
        tuple: (accuracy, precision, recall, f_score, balanced_accuracy)
    """
    match_count, label_true, label_pred = 0, [], []
    
    for image_id in set(gt_dict.keys()) | set(pred_dict.keys()):
        gt_values = gt_dict.get(image_id, {})
        pred_values = pred_dict.get(image_id, {})
        
        for key in keys:
            gt_val = gt_values.get(key, "none")
            pred_val = pred_values.get(key, "none")
            gt_val = normalize_value(gt_val)
            pred_val = normalize_value(pred_val)
            label_true.append(gt_val)
            label_pred.append(pred_val)
            if gt_val == pred_val:
                match_count += 1
    
    accuracy = match_count / len(label_true) if label_true else 0.0
    precision, recall, f_score, _ = precision_recall_fscore_support(label_true, label_pred, average='macro', zero_division=1)
    balanced_acc = balanced_accuracy_score(label_true, label_pred) if label_true and label_pred else 0.0
    
    return accuracy, precision, recall, f_score, balanced_acc

# Function to compute metrics per category
def compute_metrics_per_category(gt_dict, pred_dict, categories):
    """
    Compute evaluation metrics for each category.

    Args:
        gt_dict (dict): Ground truth captions dictionary.
        pred_dict (dict): Predicted captions dictionary.
        categories (dict): Dictionary mapping category names to list of keys.

    Returns:
        dict: Metrics per category.
    """
    return {category: compute_metrics(gt_dict, pred_dict, keys) for category, keys in categories.items()}

# Dictionary of regex patterns for keys used to insert newline before keys
key_patterns = {
    "what is the current phase?":       r"what\s+is\s+the\s+current\s*_?\s*phase\??",
    "what is the current step?":        r"what\s+is\s+the\s+current\s*_?\s*step\??",
    "what is instrument1?":             r"what\s+is\s+instrument\s*_?\s*1\??",
    "where is instrument1?":            r"where\s+is\s+instrument\s*_?\s*1\??",
    "what is instrument2?":             r"what\s+is\s+instrument\s*_?\s*2\??",
    "where is instrument2?":            r"where\s+is\s+instrument\s*_?\s*2\??",
    "what is the current action?":      r"what\s+is\s+the\s+current\s+action\??",
    "what is the next phase?":          r"what\s+is\s+the\s+next\s*_?\s*phase\??",
    "what is the next step?":           r"what\s+is\s+the\s+next\s*_?\s*step\??",
    "the number of instrument":         r"the\s+number\s+of\s+instrument"
}

# Function to insert a newline before key patterns in the caption string
def insert_newline_before_keys(caption: str, patterns: dict) -> str:
    """
    Insert a newline character before key patterns in the caption.

    Args:
        caption (str): The original caption string.
        patterns (dict): Dictionary of key and regex pattern.

    Returns:
        str: Modified caption with newline characters inserted.
    """
    for key_name, pattern in patterns.items():
        # Insert newline before the matched key pattern
        caption = re.sub(
            rf"(\b{pattern}\b)",
            r"\n\1",
            caption,
            flags=re.IGNORECASE
        )
    return caption

# Function to transform prediction JSON file captions by inserting newlines before keys
def transform_pred_json(input_json_path, output_json_path):
    """
    Transform the prediction JSON file by inserting newlines before keys in the caption.

    Args:
        input_json_path (str): Path to the input prediction JSON file.
        output_json_path (str): Path to save the transformed prediction JSON file.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # List of dictionaries with "image_id" and "caption"
    
    for item in data:
        original_caption = item.get("caption", "")
        transformed_caption = insert_newline_before_keys(original_caption, key_patterns)
        item["caption"] = transformed_caption
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Transformation complete! Saved to: {output_json_path}")

# Main function to generate JSON files (ground truth and transformed predictions) and compute metrics
def main():
    parser = argparse.ArgumentParser(description='Evaluate Captioning Model and Generate JSON Files')
    parser.add_argument('--dir', type=str, required=True, help='Base directory for dataset and predictions')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number for evaluation')
    parser.add_argument('--gt_csv', type=str, default=None, help='Path to the ground truth CSV file')
    parser.add_argument('--gt_format', type=str, default='refined_description_250225', help='Column name for caption in ground truth CSV')
    args = parser.parse_args()
    
    base_dir = args.dir
    epoch = args.epoch
    
    # Define paths for ground truth and prediction JSON files
    gt_json_path = os.path.join(base_dir, "test_pit_qa_revision_refined_data_all.json")
    pred_original_path = os.path.join(base_dir, f"test_result_epoch{epoch}.json")
    pred_revised_path = os.path.join(base_dir, f"test_result_epoch{epoch}_revision.json")
    
    # Generate ground truth JSON from CSV if --gt_csv argument is provided
    if args.gt_csv:
        if not os.path.exists(gt_json_path):
            csv_to_json(args.gt_csv, gt_json_path, format_style=args.gt_format)
    else:
        if not os.path.exists(gt_json_path):
            print(f"Ground truth JSON file '{gt_json_path}' does not exist and no CSV provided.")
            return
    
    # Transform prediction JSON file if the revised version does not exist
    if not os.path.exists(pred_revised_path):
        if os.path.exists(pred_original_path):
            transform_pred_json(pred_original_path, pred_revised_path)
        else:
            print(f"Original prediction JSON file '{pred_original_path}' does not exist.")
            return
    
    # Load ground truth and prediction data
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pred_revised_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Parse captions for ground truth and predictions
    gt_dict = {item["image_id"]: parse_caption(item["caption"]) for item in gt_data}
    pred_dict = {item["image_id"]: parse_caption(item["caption"]) for item in pred_data}
    # Define categories for evaluation metrics
    categories = {
        "phase": ["what is the current phase?", "what is the next phase?"],
        "step": ["what is the current step?", "what is the next step?"],
        "instruments": ["what is instrument1?", "what is instrument2?"],
        "positions": ["where is instrument1?", "where is instrument2?"],
        "operation_notes": ["what is the current action?"],
        "quantity": ["the number of instrument"]
    }
    
    # Compute overall metrics using all keys from categories
    overall_keys = sum(categories.values(), [])
    overall_metrics = compute_metrics(gt_dict, pred_dict, overall_keys)
    print("\n===== Overall Performance =====")
    print(f"Accuracy: {overall_metrics[0]:.4f}")
    print(f"Precision: {overall_metrics[1]:.4f}")
    print(f"Recall: {overall_metrics[2]:.4f}")
    print(f"F1-Score: {overall_metrics[3]:.4f}")
    print(f"Balanced Accuracy: {overall_metrics[4]:.4f}\n")
    
    # Compute and print metrics for each category
    category_metrics = compute_metrics_per_category(gt_dict, pred_dict, categories)
    print("===== Category-wise Performance =====")
    for category, metrics in category_metrics.items():
        print(f"[{category}] Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1-Score: {metrics[3]:.4f}, Balanced Accuracy: {metrics[4]:.4f}")

if __name__ == "__main__":
    main()
