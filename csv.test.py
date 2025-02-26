import csv
from collections import defaultdict
from typing import final

from tensorflow.python.autograph.pyct.common_transformers.anf import transform

from SplitVideo import SplitVideo
from TheCounter import TheCounter
import os
import torch
import concurrent.futures
from utility import generate_unique_string
import pandas as pd
from itertools import groupby
from datetime import datetime, timedelta

# 📂 ตั้งค่าไฟล์วิดีโอและโมเดล
video_path = "C:/Users/loh-ai/Videos/15.mp4"
model_path = "runs/detect/train14/weights/best.pt"

def folder_check():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('tmp/vdo_output'):
        os.makedirs('tmp/vdo_output')
    if not os.path.exists('tmp/split_vdo'):
        os.makedirs('tmp/split_vdo')

# ฟังก์ชันที่จะตั้งค่าให้แต่ละ process ใช้ GPU ที่แตกต่างกัน
def set_device_for_process(process_id):
    # กำหนดให้แต่ละ process ใช้ GPU ที่แตกต่างกัน
    device = torch.device(f"cuda:{process_id % torch.cuda.device_count()}")  # สลับระหว่าง GPU
    torch.cuda.set_device(device)


def process_file(file_path, process_id, name, model_path):
    # ตั้งค่า GPU สำหรับแต่ละ process
    set_device_for_process(process_id)

    # ตอนนี้เราสามารถใช้ GPU ในการประมวลผลได้
    print(f"Processing {file_path} on GPU {torch.cuda.current_device()}")
    TheCounter(name=name, video_path=file_path, model_path=model_path)


# ฟังก์ชันสำหรับการประมวลผลไฟล์ในโฟลเดอร์
def process_folder(name, model_path):
    folder_path = f"tmp/split_vdo/{name}"

    # สร้าง ProcessPoolExecutor เพื่อประมวลผลไฟล์ใน parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        # ลิสต์ไฟล์ทั้งหมดในโฟลเดอร์นี้
        for root, _, files in os.walk(folder_path):
            for process_id, file in enumerate(files):
                file_path = os.path.join(root, file)
                # ส่งงานให้แต่ละ process ใน pool
                executor.submit(process_file, file_path, process_id, name, model_path)


def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def addHour(_hr):
    # Starting time
    start_time = "18:00"
    # Convert to datetime object
    time_obj = datetime.strptime(start_time, "%H:%M")
    # Add 1 hour
    new_time = time_obj + timedelta(hours=_hr)
    # Format the new time as a string
    new_time_str = new_time.strftime("%H:%M")
    # Print the result
    return new_time_str

if __name__ == '__main__':
    csv_folder = f'tmp/vdo_output/IoRrXXLxdKa2JXAGAD'
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(csv_folder) if file.endswith(".csv")]

    obj_data = []

    # Print the list of CSV files
    _hr = 1
    for csv_file in sorted(csv_files):
        file_obj_data = []
        _file = f"{csv_folder}/{csv_file}"
        print(_file)
        data = pd.read_csv(_file)
        df = pd.read_csv(_file, header=None)  # Assuming no header
        for i in range(1, len(df)):
            file_obj_data.append({
                "name": df[0][i],
                "type": df[1][i],
                "count": df[2][i],
                "time": df[3][i],
                "hour":_hr,})
        # print(file_obj_data)

        # Group by (name)
        grouped_name = defaultdict(list)
        for item in sorted(file_obj_data, key=lambda x: (x['name'])):
            key = (item['name'])
            grouped_name[key].append(item)
        # Print the grouped result
        for key, values in grouped_name.items():
            grouped_type = defaultdict(list)
            for item in values:
                _key = (item['type'])
                grouped_type[_key].append(item)

            _in=0
            _out=0
            for __key, __values in grouped_type.items():
                if __key == 'IN':
                    _in = len(__values)
                else:
                    _out = len(__values)
            print( _hr, key ,_in,_out)
            obj_data.append([
                addHour(_hr),
                key,
                _in,
                _out,
                _in+_out
            ])
        _hr += 1

    csv_file_final = f"{csv_folder}/final.csv"
    with open(csv_file_final, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Hour", "Name", "In", "Out",'total'])
        for row in obj_data:
            writer.writerow(row)
    # Create a DataFrame
    final_data = pd.read_csv(csv_file_final)

    # Group by (Hour)
    grouped_hour = defaultdict(list)
    for item in obj_data:
        key = (item[0])
        grouped_hour[key].append(item)
    transformed_data = []
    # Print the grouped result
    for key, values in grouped_hour.items():
        #  time  bicycle  car  motorcycle  people  samlor  truck
        new_line = [key,0,0,0,0,0,0]
        for item in values:
            if item[1] == 'bicycle':
                new_line[1] = item[4]
            elif item[1] == 'car':
                new_line[2] = item[4]
            elif item[1] == 'motorcycle':
                new_line[3] = item[4]
            elif item[1] == 'people':
                new_line[4] = item[4]
            elif item[1] == 'samlor':
                new_line[5] = item[4]
            elif item[1] == 'truck':
                new_line[6] = item[4]
        transformed_data.append(new_line)
    csv_transformed = f"{csv_folder}/transformed.csv"
    with open(csv_transformed, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "bicycle", "car", "motorcycle", "people", "samlor", "truck"])
        for row in transformed_data:
            writer.writerow(row)