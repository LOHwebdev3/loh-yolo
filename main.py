import argparse
import os
import shutil
import gdown
import zipfile

from ultralytics import YOLO

def downloadFile(name,file_id):
    # สร้างโฟลเดอร์
    folder_path = "datasets/" + name
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs("download", exist_ok=True)
    os.makedirs("extract", exist_ok=True)

    # ลิงก์ของไฟล์ที่ต้องการดาวน์โหลด
    url = f'https://drive.google.com/uc?id={file_id}'

    # เส้นทางไฟล์ .zip
    zip_file_path = f'download/{name}.zip'

    # ดาวน์โหลดไฟล์
    gdown.download(url, zip_file_path, quiet=False)

    # แตกไฟล์ .zip
    extract_folder = f'extract/{name}'
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    shutil.move(f'extract/{name}/test', f'datasets/{name}/test')
    shutil.move(f'extract/{name}/train', f'datasets/{name}/train')
    shutil.move(f'extract/{name}/val', f'datasets/{name}/val')
    os.makedirs(f'{name}', exist_ok=True)
    shutil.move(f'extract/{name}/data.yaml', f'{name}/data.yaml')

    # ลบไฟล์ .zip
    os.remove(zip_file_path)

    print(f'ไฟล์ถูกแตกในโฟลเดอร์ {extract_folder}')


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--model', type=str, required=True, help='Path to pre-trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (e.g., 0 for GPU, cpu for CPU)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--fileid', type=str, required=True, help='google drive file id')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.name == 'extract' or args.name == 'datasets' or args.name == 'download':
        print("Name can't be extract, datasets, download")
        exit()
    print(args)
    downloadFile(args.name,args.fileid)


    try:
        # Load a model
        model = YOLO(args.model)


        # Train the model
        train_results = model.train(
            # data=f'extract/{args.name}/data.yaml',  # path to dataset YAML
            data=f'{args.name}/data.yaml',  # path to dataset YAML
            epochs=args.epochs,  # number of training epochs
            imgsz=640,  # training image size
            batch=16,
            device=0,1  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        )

        # Evaluate model performance on the validation set
        metrics = model.val()

        # Perform object detection on an image
        results = model(f"extract/{args.name}/sample.jpg")
        if len(results) == 0:
            print("No results returned by the model.")
        else:
            print(results)
            # Perform object detection on an image
            results[0].show()

        # Export the model to ONNX format
        path = model.export(format="onnx")  # return path to exported model
        shutil.copy(f'runs/detect/train/weights/best.pt', f'{args.name}/best.pt')
        shutil.copy(f'runs/detect/train/weights/last.pt', f'{args.name}/last.pt')

    except Exception as e:
        print(e)


    #
    # ลบโฟลเดอร์ (หากต้องการลบโฟลเดอร์ที่ไม่ใช่ไฟล์ .zip)
    shutil.rmtree(f'extract/{args.name}')
    shutil.rmtree(f'datasets/{args.name}')
    # shutil.rmtree(f'{args.name}')
    shutil.rmtree(f'runs')

