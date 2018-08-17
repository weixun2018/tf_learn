import os
import numpy as np
import tensorlayer as tl
import random

dataset_dir = '/mnt/keras-retinanet/train'

original_label_file = '/mnt/keras-retinanet/train.txt'

new_label_file = '/mnt/keras-retinanet/new_train.txt'

new_data_dict = {}
with open(original_label_file) as f:
    for item in f.readlines()[2:]:
        item_split = item.split()
        label = item_split[0].split('/')[1]
        bbox = list(map(int, item_split[2:]))
        new_data_dict[item_split[0]] = [label, bbox]

# print(['{}'.format(i) for i in new_data[:10]])
# print('{}'.format(i) for i in new_data[:10])
print('\n'.join('{}'.format(i) for i in list(new_data_dict.items())[:10]))
max_aug_fold = 17
max_aug_num = 2000

crop_fraction_list = np.arange(0.95, 0.55, -0.05)
print(crop_fraction_list)

dir_list = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
print(len(dir_list))
assert len(dir_list) == 50


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def reconvert(size, coord):
    img_w, img_h = size
    w = coord[2] * img_w
    h = coord[3] * img_h
    x_c = coord[0] * img_w
    y_c = coord[1] * img_h
    x_min = x_c - w / 2
    x_max = x_c + w / 2
    y_min = y_c - h / 2
    y_max = y_c + h / 2

    return list(map(int,[x_min, y_min, x_max, y_max]))


def augmentation(cls_dir, aug_fold):
    for img in [im for im in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, im)) and im.endswith('jpg')]:
        img_path = os.path.join(cls_dir, img)
        image = tl.vis.read_image(img_path)
        print(img_path, image.shape)
        h, w = image.shape[:2]
        cls, bbox = new_data_dict['/'.join(img_path.split('/')[3:])]
        coord = convert((w, h), bbox)

        _img = image
        _coord = coord
        for i in range(aug_fold):
            if i % 2 == 0:  # flip
                im_new, coords = tl.prepro.obj_box_left_right_flip(_img, [_coord], is_rescale=True, is_center=True,
                                                                   is_random=False)
            else:  # crop
                fraction = crop_fraction_list[int(i / 2)]
                im_new, clas, coords = tl.prepro.obj_box_crop(image, [cls],
                                                              [coord], wrg=int(w * fraction), hrg=int(h * fraction),
                                                              is_rescale=True, is_center=True, is_random=False)

                if len(clas) == 0:
                    break
                _img = im_new
                _coord = coords[0]
            # be careful!!
            bb = reconvert(im_new.shape[:2][::-1], coords[0])
            #if img == '00000036.jpg':
            #    print(img, i, im_new.shape, bbox, bb, coord, coords[0])
            img_aug_path = os.path.join(cls_dir, img.split('.')[0] + '_' + str(i) + '.jpg')
            tl.vis.save_image(im_new, img_aug_path)
            aug_data_dict['/'.join(img_aug_path.split('/')[3:])] = [cls, bb]


def clean(dataset_dir):
    for img in os.listdir(dataset_dir):
        if img.startswith('.'):
            os.remove(os.path.join(dataset_dir, img))
            continue
        if not os.path.isfile(os.path.join(dataset_dir, img)) or not img.endswith('jpg'):
            os.remove(os.path.join(dataset_dir, img))
            continue


non_empty_dir_num = 0
aug_data_dict = {}
for d in dir_list:
    dr = os.path.join(dataset_dir, d)
    # print(dr)
    clean(dr)
    img_num = len(os.listdir(dr))
    if img_num == 0:
        print('{} is empty~'.format(dr))
        continue
    non_empty_dir_num += 1
    aug_fold = int(np.ceil(max_aug_num / img_num)) - 1
    # if d == 'Onesie':
    augmentation(dr, min(aug_fold, max_aug_fold))

# count the average image num after augmentation
img_total = 0
for d in dir_list:
    dr = dr = os.path.join(dataset_dir, d)
    img_num = len(os.listdir(dr))
    img_total += img_num

print('total img:{}'.format(img_total))
print('class not empyt:{}'.format(non_empty_dir_num))
img_avg = int(img_total / non_empty_dir_num)
print('average image num:{}'.format(img_avg))

# down sample the class that image num greater than average image num
for d in dir_list:
    dr = dr = os.path.join(dataset_dir, d)
    img_num = len(os.listdir(dr))
    select_images = os.listdir(dr)
    if img_num >= img_avg:
        print('image num greater than average:{}'.format(d))
        select_images = random.sample(os.listdir(dr), img_avg)
    for img in select_images:
        img_path = os.path.join('train', d, img)
        if new_data_dict.has_key(img_path):
            aug_data_dict[img_path] = new_data_dict[img_path]

with open(new_label_file, 'w') as f:
    for item in aug_data_dict.items():
        key = item[0]
        value = item[1]
        value[1] = ' '.join(map(str, value[1]))
        s = ' '.join([key, value[0], value[1]])
        f.write(s + '\n')
