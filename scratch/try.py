import re
from pathlib import Path
from collections import defaultdict


def get_disaster_stats(root_path):
    root = Path(root_path)

    # 【关键修改】定义三级嵌套字典
    # 第一层是 event_name, 第二层是 img_id, 第三层是 status (pre/post)
    disaster_data = defaultdict(lambda: defaultdict(dict))

    # 正则提取：(事件名)_(ID)_(状态)
    pattern = re.compile(r'^(.*)_(\d+)_(pre|post)_disaster')

    # 扫描所有图片
    all_images = list(root.glob("**/images/*.jpg")) + list(root.glob("**/images/*.png"))

    for img_path in all_images:
        match = pattern.match(img_path.stem)
        if match:
            event_name = match.group(1)  # 第一级：灾害类型
            img_id = match.group(2)  # 第二级：场景ID
            status = match.group(3)  # 第三级：状态

            # 存入三级字典
            disaster_data[event_name][img_id][status] = img_path

    # --- 输出统计信息 ---
    print(f"{'灾害类型 (Event)':<30} | {'场景总数 (IDs)':<15} | {'文件总数 (Files)'}")
    print("-" * 70)

    for event, id_dict in disaster_data.items():
        # 计算该灾害类型下的文件总数
        # id_dict.values() 拿到的是一个个小的 {'pre': path, 'post': path} 字典
        total_files = sum(len(status_dict) for status_dict in id_dict.values())

        # id_dict 的长度就是该类型下有多少个不同的拍摄地点 (ID)
        total_ids = len(id_dict)

        print(f"{event:<30} | {total_ids:<15} | {total_files}")

    return disaster_data

# 使用示例
dataset_root = "dataset/EBD"
data = get_disaster_stats(dataset_root)