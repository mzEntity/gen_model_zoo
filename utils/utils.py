import os
import json
import csv
from dotenv import load_dotenv, find_dotenv

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
        

def read_csv_to_dicts(file_path):
    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


def save_dicts_to_csv(data, file_path, fieldnames=None):
    if not data:
        raise ValueError("data 不能为空")
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
