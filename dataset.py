import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from lxml import etree
from xml.etree import ElementTree


#------------------------------------------------------#
#   Reference from : https://reurl.cc/Q4EEEZ
#------------------------------------------------------#
class yoloDataset(Dataset): 
    def __init__(self, root, input_shape=[640, 640], transforms=None, dataset_property="aeroplane_train"):
        self.root = root 
        self.transforms = transforms  
        self.images_dir = os.path.join(self.root, "VOC2007/JPEGImages")  
        self.annotations_dir = os.path.join(self.root, "VOC2007/Annotations")  
        self.imagesets_dir = os.path.join(self.root, "VOC2007/ImageSets/Main")
        self.cls = dataset_property.split("_")[0]
        self.h, self.w = input_shape

        with open(os.path.join(self.imagesets_dir, f"{dataset_property}.txt")) as f: 
            file = f.readlines()
            images = np.array([x.split()[0] for x in file])
            tags = np.array([x.split()[1] for x in file])
        
        self.data_names = images[tags == '1']

        self.label_dict = { 
            "aeroplane": 1
        }

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.data_names[index] + ".jpg")
        image = Image.open(image_path).convert("RGB")  # PIL image
        iw, ih  = image.size

        annotation_path = os.path.join(self.annotations_dir, self.data_names[index] + ".xml")  # xml path
        label_names, boxes = self.read_xml(annotation_path)
        labels = np.array([self.label_dict[i] for i in label_names])

        scale = min(self.w / iw, self.h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (self.w - nw) // 2
        dy = (self.h - nh) // 2

        #-----------------------------------------------------#
        #  Add gray bars to the redundant part of the image
        #-----------------------------------------------------#
        image       = image.resize((nw, nh), Image.BICUBIC)
        new_image   = Image.new('RGB', (self.w, self.h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data  = np.transpose(np.array(new_image), (2, 0, 1))

        #-----------------------------------------------------#
        #   Make adjustments to the ground truth 
        #-----------------------------------------------------#
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw/iw + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh/ih + dy
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > self.w] = self.w
            boxes[:, 3][boxes[:, 3] > self.h] = self.h
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)] # discard invalid box

        labels, boxes = map(lambda t: torch.as_tensor(t), [labels, boxes])

        target = { 
            "boxes": boxes,  
            "labels": labels  
        }

        if self.transforms is not None:  
            image, target = self.transforms(image, target)

        return image_data, target 

    def read_xml(self, annotation_path):  # Read xml
        objnames = []  # Name
        objboxes = []  # Boxs

        parser = etree.XMLParser(encoding="utf-8")
        xmlroot = ElementTree.parse(annotation_path, parser=parser).getroot()
        for object in xmlroot.findall("object"):
            if(object.find("name").text != self.cls):   #  Restriction Class Type
                continue
            objnames.append(object.find("name").text)
            objxmin = float(object.find("bndbox/xmin").text)
            objymin = float(object.find("bndbox/ymin").text)
            objxmax = float(object.find("bndbox/xmax").text)
            objymax = float(object.find("bndbox/ymax").text)
            assert objxmax > objxmin and objymax > objymin  
            objboxes.append([objxmin, objymin, objxmax, objymax])

        return np.array(objnames), np.array(objboxes)
