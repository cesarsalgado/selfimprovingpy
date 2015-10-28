import numpy as np
import os
from os.path import join
import re

TINY_IMAGENET_PATH = '/media/cesar/Acer/Users/cesarsalgado/datasets/\
                      tiny-imagenet-200'


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def write_label_names(label_names):
    with open(join(TINY_IMAGENET_PATH, 'label_names.txt'), 'w') as f:
        f.write('\n'.join(label_names))


def read_label_names():
    with open(join(TINY_IMAGENET_PATH, 'label_names.txt'), 'r') as f:
        label_names = f.read().splitlines()
    return label_names


def make_elem_to_idx(list_):
    elem_to_idx = {}
    for i, elem in enumerate(list_):
        elem_to_idx[elem] = i
    return elem_to_idx


def gen_train_img_paths_and_labels_txts():
    train_path = join(TINY_IMAGENET_PATH, 'train')
    label_names = sorted(next(os.walk(train_path))[1])
    label_name_to_int = make_elem_to_idx(label_names)
    int_labels = []
    boxes = []
    with open(join(train_path, 'paths.txt'), 'w') as wf:
        for label_name in label_names:
            label_path = join(train_path, label_name)
            with open(join(label_path, label_name + '_boxes.txt'), 'r') as f:
                imgs_boxes = f.readlines()
            label_imgs_path = join(label_path, 'images')
            for img_box in imgs_boxes:
                int_labels.append(label_name_to_int[label_name])
                parts = img_box.split()
                img_path = join(label_imgs_path, parts[0])
                wf.write(img_path + '\n')
                boxes.append(map(int, parts[1:]))
    np.savetxt(join(train_path, 'labels.txt'), int_labels, '%d')
    np.savetxt(join(train_path, 'boxes.txt'), np.array(boxes), '%d')
    write_label_names(label_names)


# need to run gen_train_img_paths_and_labels_txts before
def gen_val_img_paths_and_labels_txts():
    val_path = join(TINY_IMAGENET_PATH, 'val')
    with open(join(val_path, 'val_annotations.txt'), 'r') as f:
        lines = f.readlines()
    label_names = read_label_names()
    label_name_to_int = make_elem_to_idx(label_names)
    imgs_path = join(val_path, 'images')
    int_labels = []
    boxes = []
    with open(join(val_path, 'paths.txt'), 'w') as f:
        for line in lines:
            parts = line.split()
            img_path = join(imgs_path, parts[0])
            f.write(img_path + '\n')
            int_labels.append(label_name_to_int[parts[1]])
            boxes.append(map(int, parts[2:]))
    np.savetxt(join(val_path, 'labels.txt'), int_labels, '%d')
    np.savetxt(join(val_path, 'boxes.txt'), np.array(boxes), '%d')


def gen_test_img_paths():
    test_path = join(TINY_IMAGENET_PATH, 'test')
    test_imgs_path = join(test_path, 'images')
    imgs_names = os.listdir(test_imgs_path)
    sort_nicely(imgs_names)
    paths = [join(test_imgs_path, name) for name in imgs_names]
    with open(join(test_path, 'paths.txt'), 'w') as f:
        f.write('\n'.join(paths))
