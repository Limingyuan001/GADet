import os
import xml.etree.ElementTree as ET
from PIL import Image
#
# data_dir = "E:\Datasets\OPIXray\OPIXray"
# label_dir = f"{data_dir}/train/train_annotation"
# output_dir = "E:\Datasets\OPIXray\OPIXray_voc"
# image_dir = f"{output_dir}/JPEGImages"
# annotation_dir = f"{output_dir}/Annotations"



import os
import xml.etree.ElementTree as ET
from PIL import Image

data_dir = "E:/Datasets/OPIXray/OPIXray"
train_image_dir = f"{data_dir}/train/train_image"
train_label_dir = f"{data_dir}/train/train_annotation"
test_image_dir = f"{data_dir}/test/test_image"
test_label_dir = f"{data_dir}/test/test_annotation"
output_dir = "E:\Datasets\OPIXray\OPIXray_voc"
image_dir = f"{output_dir}/JPEGImages"
annotation_dir = f"{output_dir}/Annotations"

# 获取训练集图像和标签文件路径
train_image_paths = [f"{train_image_dir}/{filename}" for filename in os.listdir(train_image_dir)]
train_label_paths = [f"{train_label_dir}/{os.path.splitext(filename)[0]}.txt" for filename in os.listdir(train_image_dir)]

# 获取测试集图像和标签文件路径
test_image_paths = [f"{test_image_dir}/{filename}" for filename in os.listdir(test_image_dir)]
test_label_paths = [f"{test_label_dir}/{os.path.splitext(filename)[0]}.txt" for filename in os.listdir(test_image_dir)]

# 将训练集和测试集文件路径合并
image_paths = train_image_paths + test_image_paths
label_paths = train_label_paths + test_label_paths

for image_path, label_path in zip(image_paths, label_paths):
    # 从标签文件中读取类别标签和边界框坐标
    with open(label_path, "r") as f:
        filename_, class_label, x_min, y_min, x_max, y_max = f.readline().split()

    # 创建输出目录
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    # 复制图像到输出目录
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    image.save(f"{image_dir}/{image_name}")

    # 创建注释XML文件
    annotation_name = os.path.splitext(image_name)[0] + ".xml"
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.width)
    ET.SubElement(size, "height").text = str(image.height)
    object = ET.SubElement(root, "object")
    ET.SubElement(object, "name").text = class_label
    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = x_min
    ET.SubElement(bndbox, "ymin").text = y_min
    ET.SubElement(bndbox, "xmax").text = x_max
    ET.SubElement(bndbox, "ymax").text = y_max
    tree = ET.ElementTree(root)
    tree.write(f"{annotation_dir}/{annotation_name}")


# convert txt files to xml files
# for txt_file in os.listdir('E:/Datasets/OPIXray/OPIXray/train/train_annotation'):
#     img_name = os.path.splitext(txt_file)[0] + '.jpg'
#     xml_file = os.path.splitext(txt_file)[0] + '.xml'
#     xml_path = os.path.join('Annotations', xml_file)
#     with open(os.path.join('E:/Datasets/OPIXray/OPIXray/train/train_annotation', txt_file), 'r') as f:
#         data = f.readline().strip().split(' ')
#         label = data[1]
#         x1, y1, x2, y2 = data[2:]
#     root = ET.Element('annotation')
#     folder = ET.SubElement(root, 'folder')
#     folder.text = 'JPEGImages'
#     filename = ET.SubElement(root, 'filename')
#     filename.text = img_name
#     path = ET.SubElement(root, 'path')
#     path.text = os.path.join('E:/Datasets/OPIXray/OPIXray/train/train_image', img_name)
#     source = ET.SubElement(root, 'source')
#     database = ET.SubElement(source, 'database')
#     database.text = 'Unknown'
#     size = ET.SubElement(root, 'size')
#     width = ET.SubElement(size, 'width')
#     width.text = '512'  # replace with actual image width
#     height = ET.SubElement(size, 'height')
#     height.text = '512'  # replace with actual image height
#     depth = ET.SubElement(size, 'depth')
#     depth.text = '3'
#     segmented = ET.SubElement(root, 'segmented')
#     segmented.text = '0'
#     obj = ET.SubElement(root, 'object')
#     name = ET.SubElement(obj, 'name')
#     name.text = label
#     pose = ET.SubElement(obj, 'pose')
#     pose.text = 'Unspecified'
#     truncated = ET.SubElement(obj, 'truncated')
#     truncated.text = '0'
#     difficult = ET.SubElement(obj, 'difficult')
#     difficult.text = '0'
#     bndbox = ET.SubElement(obj, 'bndbox')
#     xmin = ET.SubElement(bndbox, 'xmin')
#     xmin.text = x1
#     ymin = ET.SubElement(bndbox, 'ymin')
#     ymin.text = y1
#     xmax = ET.SubElement(bndbox, 'xmax')
#     xmax.text = x2
#     ymax = ET.SubElement(bndbox, 'ymax')
#     ymax.text = y2
#     tree = ET.ElementTree(root)
#     tree.write(xml_path)