import os
import json
import csv
from dotenv import load_dotenv, find_dotenv
import yaml


def setup():
    while os.getenv('ENV_LOADED') is None:
        _ = load_dotenv(find_dotenv(), override=True)
    print("setup over")
    
    
    
def read_json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_dict_to_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
        
        
def read_yaml_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
    
def save_dict_to_yaml(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_csv_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


def save_dict_to_csv(data, file_path, fieldnames=None):
    if not data:
        raise ValueError("data 不能为空")
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
        
EXTENSION_MAP = {
    '.json': (read_json_to_dict, save_dict_to_json),
    '.yaml': (read_yaml_to_dict, save_dict_to_yaml),
    '.yml':  (read_yaml_to_dict, save_dict_to_yaml),
    '.csv':  (read_csv_to_dict, save_dict_to_csv)
}

def read_file_to_dict(file_path, *args, **kwargs):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in EXTENSION_MAP:
        raise ValueError(f"不支持的文件格式: {ext}")
    reader, _ = EXTENSION_MAP[ext]
    return reader(file_path, *args, **kwargs)

def save_dict_to_file(data, file_path, *args, **kwargs):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in EXTENSION_MAP:
        raise ValueError(f"不支持的文件格式: {ext}")
    _, writer = EXTENSION_MAP[ext]
    return writer(data, file_path, *args, **kwargs)