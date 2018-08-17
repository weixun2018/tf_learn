import tensorlayer as tl

# 下载 VOC 2012 数据集
imgs_file_list, _, _, _, classes, _, _, \
_, objs_info_list, _ = tl.files.load_voc_dataset(dataset="2012")

# 图片标记预处理为列表形式
ann_list = []
for info in objs_info_list:
    ann = tl.prepro.parse_darknet_ann_str_to_list(info)
    c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    ann_list.append([c, b])

# 读取一张图片，并保存
idx = 2  # 可自行选择图片
image = tl.vis.read_image(imgs_file_list[idx])
tl.vis.draw_boxes_and_labels_to_image(image, ann_list[idx][0],
                                      ann_list[idx][1], [], classes, True, save_name='_im_original.png')
