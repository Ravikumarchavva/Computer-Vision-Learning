import torch
import os
import pandas as pd
from PIL import Image


class YOLODataset:
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=448, S=7, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.C = C
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        target = torch.zeros((self.S, self.S, self.C + 5 * len(boxes)))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if target[i, j, 20] == 0:
                target[i, j, class_label] = 1
                target[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                target[i, j, 21:25] = box_coordinates
                target[i, j, class_label] = 1

        return image, target
    
from torchvision.transforms import Compose, Resize, ToTensor


if __name__ == "__main__":
    IMAGE_DIR = "../data/JPEGImages"
    LABEL_DIR = "../data/Annotations"
    anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
    transform = Compose([Resize((448, 448)), ToTensor()])
    dataset = YOLODataset(
        "data/labels/train.csv", IMAGE_DIR, LABEL_DIR, anchors, transform=transform
    )
    img, target = dataset[0]
    print(img.shape, target.shape)
    print(target)