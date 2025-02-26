import pandas as pd
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('tmp/vdo_output/eIn1SySGhhW47zhoT5/final.csv')

# กลุ่มข้อมูลตาม 'Hour', 'name', 'Type' และคำนวณจำนวนรวม
grouped_data = data.groupby(['Hour', 'Name', 'Type'])['Count'].sum().unstack(fill_value=0)

# สร้างกราฟแท่ง
fig, ax = plt.subplots(figsize=(12, 8))

# แสดงกราฟแท่งแยกตาม 'Hour' และ 'name'
grouped_data.plot(kind='bar', stacked=False, ax=ax)

# ตั้งชื่อกราฟและแกน
ax.set_title("Vehicle Counts by Hour, Name, and Type")
ax.set_xlabel("Hour")
ax.set_ylabel("Vehicle Count")
ax.set_xticklabels(grouped_data.index, rotation=0)

# เพิ่มคำอธิบายให้กับชื่อที่แสดงบนแท่งกราฟ
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 1,  # ตำแหน่งของข้อความ
            f'{height:.0f}', ha='center', va='bottom')  # เพิ่มข้อความเป็นตัวเลข

# แสดงกราฟ
plt.legend(title="Movement Type")
plt.tight_layout()  # เพื่อให้กราฟดูได้ดีขึ้น
plt.show()
