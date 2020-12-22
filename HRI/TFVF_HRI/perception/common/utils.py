import io
import cv2
import numpy as np
import pycocotools.mask as coco_mask_util
from paddle.fluid.core import PaddleTensor

from config import YOLOv3_img_resize_to, YOLOv4_img_resize_to

##################################################
#              CV related utils
##################################################


def LodTensor_to_Tensor(lod_tensor, lod=None):
    if lod is None and isinstance(lod_tensor, PaddleTensor):
        lod = lod_tensor.lod
        array = lod_tensor.as_ndarray()
    elif lod is None:
        lod = lod_tensor.lod()
        array = np.array(lod_tensor)
    else:
        array = np.array(lod_tensor)

    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(array[lod[0][i]:lod[0][i + 1]])
    return new_array


def _color_image_to_mode(img, mode='bgr'):
    assert type(mode) is str
    assert len(img.shape) == 3 and img.shape[2] == 3, \
        'Only support color image in HWC.'
    if isinstance(img, np.ndarray):
        # read image using `cv2.imread()`, default bgr mode
        bgr_mode = True
        img = np.copy(img)
    else:
        # read image using `PIL.Image.open()`, default rgb mode
        bgr_mode = False
        img = np.array(img)

    if mode.lower() == 'bgr' and not bgr_mode:
        img = img[:, :, ::-1]
    elif mode.lower() == 'rgb' and bgr_mode:
        img = img[:, :, ::-1]
    elif mode.lower() not in ['rgb', 'bgr']:
        raise ValueError('Invalid image mode: %s' % mode)

    # return a copy of transformed image
    return img


def _hwc_to_chw(img):
    assert len(img.shape) == 3 and img.shape[2] == 3, \
        'Only support color image in HWC.'
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 1, 0)
    return img


def robot2_frame_crop_resize(frame):
    h, w, _ = frame.shape
    if h == 480 and w == 640:
        # It's robot v1
        return frame
    # It's h: 720; w: 1280
    d = (w - h // 3 * 4) // 2
    crop = np.copy(frame[:, d:-d])
    return cv2.resize(crop, (640, 480))


def ppdet_img_preprocess(img,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         resize_to=None,
                         resize_interp=cv2.INTER_LINEAR,
                         to_rgb=True,
                         is_scale=True,
                         channel_first=True):
    if to_rgb:
        img = _color_image_to_mode(img, mode='rgb')
    else:
        img = _color_image_to_mode(img, mode='bgr')

    if resize_to is not None:
        im_scale_x = float(resize_to) / float(img.shape[1])
        im_scale_y = float(resize_to) / float(img.shape[0])
        img = cv2.resize(img, None, None, fx=im_scale_x,
                         fy=im_scale_y, interpolation=resize_interp)

    if is_scale:
        img = img.astype(np.float32) / 255.

    if channel_first:
        img = _hwc_to_chw(img)
        img -= np.array(mean)[:, np.newaxis, np.newaxis].astype(np.float32)
        img /= np.array(std)[:, np.newaxis, np.newaxis].astype(np.float32)
    else:
        img -= np.array(mean)[np.newaxis, np.newaxis, :].astype(np.float32)
        img /= np.array(std)[np.newaxis, np.newaxis, :].astype(np.float32)

    return img


def yolov3_img_preprocess(img,
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225],
                          resize_to=YOLOv3_img_resize_to,
                          resize_interp=cv2.INTER_CUBIC,
                          to_rgb=True,
                          is_scale=True):
    return ppdet_img_preprocess(
        img, mean=mean, std=std, to_rgb=to_rgb,
        resize_to=resize_to, resize_interp=resize_interp,
        is_scale=is_scale, channel_first=True)


def yolov4_img_preprocess(img,
                          bg_fill=[128, 128, 128],
                          resize_to=YOLOv4_img_resize_to):
    aspect_ratio = min(resize_to * 1.0 / img.shape[0],
                       resize_to * 1.0 / img.shape[1])
    new_h = int(img.shape[0] * aspect_ratio)
    new_w = int(img.shape[1] * aspect_ratio)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Generate canvas with size (resize_to, resize_to)
    boxed_img = np.zeros((resize_to, resize_to, 3)).astype(np.uint8)
    boxed_img[:, :] = np.array(bg_fill).astype(np.uint8)

    # Paste resized image
    y_offset = (resize_to - new_h) // 2
    x_offset = (resize_to - new_w) // 2
    boxed_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    # import ipdb; ipdb.set_trace()
    # cv2.imwrite('boxed_img.jpg', boxed_img)

    # Use RGB mode and rescale
    boxed_img = boxed_img[:, :, ::-1]
    boxed_img = boxed_img.astype(np.float32) / 255.

    boxed_img = _hwc_to_chw(boxed_img)
    return boxed_img


def inst_crop_preprocess(img, crop_shape):
    img = cv2.resize(np.copy(img), crop_shape)

    # Use RGB mode and rescale
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.

    img = _hwc_to_chw(img)
    return img


def yolov3_instance_postprocess(bbox, confidence_th, im_h, im_w, fm=None,
                                img=None):
    if bbox.shape[1] < 2:
        # When not found instances, bbox = [[-1.]]
        return []

    keep_index = np.where(bbox[:, 1] >= confidence_th)[0]
    bbox = bbox[keep_index, :]

    bbox[:, 2] = np.maximum(0, bbox[:, 2])
    bbox[:, 3] = np.maximum(0, bbox[:, 3])
    bbox[:, 4] = np.minimum(im_w - 1, bbox[:, 4])
    bbox[:, 5] = np.minimum(im_h - 1, bbox[:, 5])

    if fm is not None:
        fm = fm[keep_index, :, :, :]

    instances = []
    for i in range(bbox.shape[0]):
        cid, score = bbox[i, :2]
        cid = int(cid)
        cname = coco17_cid2name(cid, with_bg=False)

        instance_dict = {
            'cid': cid,
            'category': cname,
            'score': score,
            'bbox': bbox[i, 2:]
        }

        if fm is not None:
            instance_dict['fm'] = fm[i, :, :, :]

        if img is not None:
            xmin, ymin, xmax, ymax = bbox[i, 2:]
            instance_dict['crop'] = img[int(ymin):int(ymax),
                                        int(xmin):int(xmax)]

        instances.append(instance_dict)
    return instances


def coco17_cid2name(class_id, with_bg=False):
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    if not with_bg:
        return class_names[class_id]

    if class_id == 0:
        return 'background'
    else:
        return class_names[class_id-1]


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.

    Args:
        boxes (np.ndarray): in shape [N, 4], batch of xmin, ymin, xmax, ymax
        scale (float): the scale factor to expand the boxes
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def get_bbox_pos_emb(bbox, im_h, im_w, emb_h, emb_w, mode='sin'):
    """
    Get positional embedding related to the image center.
    """
    assert mode in ['sin', 'linear']
    xmin, ymin, xmax, ymax = bbox
    xmin = (xmin - im_w / 2.) / (im_w / 2.)
    ymin = (ymin - im_h / 2.) / (im_h / 2.)
    xmax = (xmax - im_w / 2.) / (im_w / 2.)
    ymax = (ymax - im_h / 2.) / (im_h / 2.)

    if mode == 'sin':
        xmin *= np.pi / 2
        ymin *= np.pi / 2
        xmax *= np.pi / 2
        ymax *= np.pi / 2

    x_pos = np.linspace(xmin, xmax, num=emb_w)
    y_pos = np.linspace(ymin, ymax, num=emb_h)
    if mode == 'sin':
        x_pos = np.sin(x_pos)
        y_pos = np.sin(y_pos)

    y_pos_emb, x_pos_emb = np.meshgrid(y_pos, x_pos, indexing='ij')
    x_pos_emb = np.expand_dims(x_pos_emb, axis=0)
    y_pos_emb = np.expand_dims(y_pos_emb, axis=0)
    pos_emb = np.concatenate((x_pos_emb, y_pos_emb), axis=0)
    return pos_emb


##################################################
#              NLP related utils
##################################################


def load_vocab(file_path, with_unk0=True, word_first=True, value_to_int=True):
    """
    Load the given vocabulary with kwargs to support different vocab format.
    """
    vocab = dict()
    f = io.open(file_path, 'r', encoding='utf-8')
    for num, line in enumerate(f):
        items = line.strip('\n').split('\t')
        if len(items) == 1:
            k, v = num, items[0]
        elif len(items) == 2:
            if word_first:
                k, v = items[0], items[1]
            else:
                k, v = items[1], items[0]
        else:
            raise RuntimeError('Failed to parse vocabulary file')

        if value_to_int:
            v = int(v)

        if k not in vocab:
            vocab[k] = v

    if with_unk0:
        vocab['<unk>'] = 0
    return vocab


def cosine_sim(vector_0, vector_1):
    norm_0 = np.linalg.norm(vector_0)
    norm_1 = np.linalg.norm(vector_1)
    return np.dot(vector_0, vector_1) / (norm_0 * norm_1)
