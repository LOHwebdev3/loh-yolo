import csv
import os
import cv2
import math
import logging

from ObjectCounter import ObjectCounter
from utility import to_snake_case, generate_unique_string


def TheCounter(name='', video_path='', model_path='', _angle = 30, _region_points=None,_show=False):
    if name == '':
        name = to_snake_case("ObjectCounter")

    unique_str = generate_unique_string(18)
    name = f"{name}"

    # 🛑 Suppress warnings
    if _region_points is None:
        _region_points = []
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    # 📌 เปิดวิดีโอ
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # 🔄 หมุนและสร้างสี่เหลี่ยมแทนเส้น
    angle = _angle  # มุมเอียง (องศา)
    angle_rad = math.radians(angle)

    rect_width = w*1.5  # ความกว้างของสี่เหลี่ยม
    rect_height = 150  # ความสูงของสี่เหลี่ยม
    x_center, y_center = w // 2, h // 2  # จุดศูนย์กลางของสี่เหลี่ยม

    # คำนวณจุด 4 มุมของสี่เหลี่ยมโดยใช้การหมุน
    x_offset = rect_width // 2
    y_offset = rect_height // 2

    corners = [
        (int(x_center + x_offset * math.cos(angle_rad) - y_offset * math.sin(angle_rad)),
         int(y_center + x_offset * math.sin(angle_rad) + y_offset * math.cos(angle_rad))),

        (int(x_center - x_offset * math.cos(angle_rad) - y_offset * math.sin(angle_rad)),
         int(y_center - x_offset * math.sin(angle_rad) + y_offset * math.cos(angle_rad))),

        (int(x_center - x_offset * math.cos(angle_rad) + y_offset * math.sin(angle_rad)),
         int(y_center - x_offset * math.sin(angle_rad) - y_offset * math.cos(angle_rad))),

        (int(x_center + x_offset * math.cos(angle_rad) + y_offset * math.sin(angle_rad)),
         int(y_center + x_offset * math.sin(angle_rad) - y_offset * math.cos(angle_rad)))
    ]


    if len(_region_points) == 0:
        region_points = corners  # กำหนดเส้นที่หมุนแล้ว
    else:
        region_points = _region_points

    file_name = os.path.basename(video_path)

    if not os.path.exists(f'tmp/vdo_output/{name}'):
        os.makedirs(f'tmp/vdo_output/{name}')
    # 🎥 ตั้งค่า Video Writer
    video_writer = cv2.VideoWriter(f"tmp/vdo_output/{name}/{file_name}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


    csv_file = f"tmp/vdo_output/{name}/{file_name}.csv"
    with open(csv_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Type", "Count", "Time (seconds)"])
        pass

    def onAddCount(name, type, count, time):
        # แสดงข้อความใน console
        # print(f"{name} | obj : {type} count: {count} at time: {time} seconds")

        # เปิดไฟล์ CSV ด้วยโหมด append
        with open(csv_file, mode="a", newline='') as file:
            writer = csv.writer(file)
            # เขียนข้อมูลใหม่ในแถว
            writer.writerow([name, type, count, round(time, 3)])

    # 🔍 ตั้งค่า ObjectCounter
    counter = ObjectCounter(
        show=_show,  # ไม่ต้องแสดง GUI
        region=region_points,
        model=model_path,
        # show_in=True,
        # show_out=True,
        # line_width=2,
        verbose=False,  # ปิดข้อความ log
        onAddCount=onAddCount  # ฟังก์ชันที่จะเรียกเมื่อ count เปลี่ยน
    )

    # 🎬 เริ่มประมวลผลวิดีโอ
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(f"📌{file_name} Video frame is empty or processing has completed.")
            break

        # Get the current timestamp in milliseconds
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds

        # 📌 นับวัตถุ
        im0 = counter.count(im0, current_time)

        # บันทึกวิดีโอที่ประมวลผลแล้ว
        video_writer.write(im0)

        # # แสดงผล (กด 'q' เพื่อออก)
        # cv2.imshow("Output", im0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 🔚 ปิดทุกอย่างหลังจากเสร็จสิ้น
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()