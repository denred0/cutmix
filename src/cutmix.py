import torch
import random
import numpy as np
import pandas as pd
import cv2
import shutil

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from tqdm import tqdm
from pathlib import Path

TRAIN_ROOT_PATH = 'data/input/train'


class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.test = test

    def __getitem__(self, index: int):
        index = self.image_ids[index]
        # index = 403
        image_name = self.marking.iloc[index]['image_name']

        if True:
            #     image, boxes = self.load_image_and_boxes(index)
            # else:
            image, boxes = self.load_cutmix_image_and_boxes(index, type='four_images')

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index])}

        return image, target, image_name

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        # print('index', index)
        # index = 403
        image_name = self.marking['image_name'][index]
        image = cv2.imread(TRAIN_ROOT_PATH + '/' + image_name + '.png', cv2.IMREAD_COLOR).astype(np.float32)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        row = self.marking.loc[index]

        #         boxes = records[['x', 'y', 'w', 'h']].values
        #         boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        #         boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        bboxes = []
        for bbox in row['BoxesString'].split(';'):
            bboxes.append(list(map(float, bbox.split(' '))))
        return image, np.array(bboxes)

    def load_cutmix_image_and_boxes(self, index, type='four_images', imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        itog_images = []
        itog_bboxes = []
        if type == 'four_images':

            for i in range(4):
                w, h = imsize, imsize
                s = imsize // 2

                xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
                indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

                result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
                result_boxes = []

                for i, index in enumerate(indexes):
                    image, boxes = self.load_image_and_boxes(index)
                    if i == 0:
                        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                                 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                                y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                    elif i == 1:  # top right
                        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                    elif i == 2:  # bottom left
                        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                    elif i == 3:  # bottom right
                        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                    result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
                    padw = x1a - x1b
                    padh = y1a - y1b

                    boxes[:, 0] += padw
                    boxes[:, 1] += padh
                    boxes[:, 2] += padw
                    boxes[:, 3] += padh

                    result_boxes.append(boxes)

                result_boxes = np.concatenate(result_boxes, 0)
                np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
                result_boxes = result_boxes.astype(np.int32)
                result_boxes = result_boxes[
                    np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]

                return result_image, result_boxes

        elif type == 'eight_images':
            for i in range(4):
                w, h = imsize, imsize
                s = imsize // 2

                xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
                indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

                result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
                result_boxes = []

                for i, index in enumerate(indexes):
                    image, boxes = self.load_image_and_boxes(index)
                    if i == 0:
                        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                                 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                                y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                    elif i == 1:  # top right
                        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                    elif i == 2:  # bottom left
                        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                    elif i == 3:  # bottom right
                        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                    result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
                    padw = x1a - x1b
                    padh = y1a - y1b

                    boxes[:, 0] += padw
                    boxes[:, 1] += padh
                    boxes[:, 2] += padw
                    boxes[:, 3] += padh

                    result_boxes.append(boxes)

                result_boxes = np.concatenate(result_boxes, 0)
                np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
                result_boxes = result_boxes.astype(np.int32)
                result_boxes = result_boxes[
                    np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]

                itog_bboxes.append(result_boxes)
                itog_images.append(result_image)

            result_image2 = np.full((imsize * 2, imsize * 2, 3), 1, dtype=np.float32)
            all_bboxes = [0, 0, 0, 0]

            result_image2[0:imsize, 0:imsize] = itog_images[0]
            all_bboxes = np.vstack([all_bboxes, itog_bboxes[0]])

            all_bboxes = all_bboxes[1:]

            result_image2[0:imsize, imsize:imsize * 2] = itog_images[1]
            for box in itog_bboxes[1]:
                box[0] += imsize
                box[2] += imsize
            all_bboxes = np.vstack([all_bboxes, itog_bboxes[1]])

            result_image2[imsize:imsize * 2, 0:imsize] = itog_images[2]
            for box in itog_bboxes[2]:
                box[1] += imsize
                box[3] += imsize
            all_bboxes = np.vstack([all_bboxes, itog_bboxes[2]])

            result_image2[imsize:imsize * 2, imsize:imsize * 2] = itog_images[3]
            for box in itog_bboxes[3]:
                box[0] += imsize
                box[1] += imsize
                box[2] += imsize
                box[3] += imsize
            all_bboxes = np.vstack([all_bboxes, itog_bboxes[3]])

            all_boxes_itog = []

            for box in all_bboxes:
                area_x = int(box[2]) - int(box[0])
                area_y = int(box[3]) - int(box[1])

                if area_x > 10 and area_y > 10:
                    all_boxes_itog.append(box)

            return result_image2, all_bboxes


def create_cutmix():
    # clear folder
    dirpath = Path('data/output')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    dirpath = Path('data/output_draw')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv('data/input/train.csv')
    df.drop(df[df.BoxesString == 'no_box'].index, inplace=True)
    df.drop(df.index[[7, 16]], inplace=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_folds = df[['image_name']].copy()
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df.image_name, y=df['domain'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    fold_number = 1

    df = df.reset_index(drop=True)

    # a = df['image_name'][403]

    train_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        marking=df,
        test=False,
    )

    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=0,
        # collate_fn=collate_fn,
    )

    image_size = 1024

    for step, (image, target, image_id) in tqdm(enumerate(train_loader)):
        image = image.squeeze(0).detach().cpu().numpy() * 255
        image = cv2.resize(image, (image_size, image_size))
        cv2.imwrite('data/output/' + str(step) + '.png', image)
        boxes_list = target['boxes'].detach().cpu().numpy()[0]
        # print()
        txt_list = []
        for boxes in boxes_list:
            x_center = ((boxes[2] - boxes[0]) / 2 + boxes[0]) / image_size
            y_center = ((boxes[3] - boxes[1]) / 2 + boxes[1]) / image_size

            w = (boxes[2] - boxes[0]) / image_size
            h = (boxes[3] - boxes[1]) / image_size

            cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 255, 0), 2)

            txt_str = '0 ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h)

            txt_list.append(txt_str)
        cv2.imwrite('data/output_draw/' + str(step) + '.png', image)

        with open('data/output/' + str(step) + '.txt', 'w') as f:
            for item in txt_list:
                f.write("%s\n" % item)

        # print('images', images.shape)


if __name__ == '__main__':
    create_cutmix()
