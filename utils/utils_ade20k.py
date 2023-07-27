from PIL import Image
import matplotlib._color_data as mcd
import cv2
import json
import numpy as np
import os

_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'


def rgb(triplet):
    """
    이 함수는 16진수 색상 코드를 RGB 값으로 변환하는 함수입니다.
    이 함수는 주어진 색상 코드를 각각의 R, G, B 성분으로 분리하고, 이를 10진수로 변환합니다."""
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]


def loadAde20K(file: str):
    """ 이 함수는 ADE20K 데이터셋에서 이미지를 불러오고, segmentation 정보를 읽어들이는 함수입니다.

    주어진 이미지 파일의 segmentation 버전을 불러옵니다.
    (ADE20K에서는 이 파일이 '_seg.png'로 끝납니다)
    segmentation 이미지에서 각각의 R, G, B 채널을 분리합니다.
        R과 G 채널을 이용해 segmentation mask를 만듭니다.
        B 채널을 이용해 instance mask를 만듭니다.
    'parts{}.png' 형식의 파일이 있는 경우, 이를 불러와서 해당 부분의 segmentation mask를 만듭니다.
    '.json' 형식의 파일이 있는 경우, 이를 불러와서 객체와 부분의 정보를 저장합니다.


    Args:
        file:

    Returns:
        {
        'img_name': file,
/Users/user/Downloads//ADE20K_2021_17_01/images/ADE/training/urban/street/ADE_train_00016869.jpg
        'segm_name': fileseg,
/Users/user/Downloads//ADE20K_2021_17_01/images/ADE/training/urban/street/ADE_train_00016869_seg.png
        'class_mask': ObjectClassMasks,
            np.ndarray: (height, width)
        'instance_mask': ObjectInstanceMasks,
            np.ndarray: (height, width)
        'partclass_mask': PartsClassMasks,
            : List[np.ndarray] (length 2)
                (1536, 2048)
                (1536, 2048)

        'part_instance_mask': PartsInstanceMasks,
            : List[np.ndarray] (length 2)
                (1536, 2048)
                (1536, 2048)
        'objects': objects,
            : Dict
                instancendx: np.array (51,)
                class: List[str]
                corrected_raw_name: List[str]
                iscrop: np.array (51,)
                listattributes: List[Union[List,str]]
                polygon: List[Dict]
        'parts': parts
            : Dict
                instancendx: np.array (19,)
                class: List[str]
                corrected_raw_name: List[str]
                iscrop: np.array (19,)
                listattributes: List[Union[List,str]]
                polygon: List[Dict]
        }
    """

    fileseg = file.replace('.jpg', '_seg.png')
    with Image.open(fileseg) as io:
        seg = np.array(io)

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:, :, 0]
    G = seg[:, :, 1]
    B = seg[:, :, 2]
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat

    level = 0
    PartsClassMasks = []
    PartsInstanceMasks = []
    while True:
        level = level + 1
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level))
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io)
            R = partsseg[:, :, 0]
            G = partsseg[:, :, 1]
            B = partsseg[:, :, 2]
            PartsClassMasks.append((np.int32(R) / 10) * 256 + np.int32(G))
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks


        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name = [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p > 0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in
                                         list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in
                                     list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in
                              list(np.where(ispart == 0)[0])]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in
                                       list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in
                                   list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'class_mask': ObjectClassMasks,
            'instance_mask': ObjectInstanceMasks,
            'partclass_mask': PartsClassMasks,
            'part_instance_mask': PartsInstanceMasks,
            'objects': objects, 'parts': parts}


def plot_polygon(img_name, info, show_obj=True, show_parts=False):
    """ 이 함수는 주어진 이미지에 polygon을 그리는 함수입니다. polygon은 객체나 부분의 외곽선을 나타냅니다.

    CSS4 색상을 불러옵니다.
    객체와 부분의 정보를 불러옵니다.
    주어진 이미지를 불러옵니다.
    각각의 객체와 부분에 대해, 해당 polygon을 그립니다.
        이 때, 색상은 CSS4 색상 중에서 순차적으로 선택됩니다.

    Args:
        img_name:
        info:
        show_obj:
        show_parts:

    Returns:

    """
    colors = mcd.CSS4_COLORS
    color_keys = list(colors.keys())
    all_objects = []
    all_poly = []
    if show_obj:
        all_objects += info['objects']['class']
        all_poly += info['objects']['polygon']
    if show_parts:
        all_objects += info['parts']['class']
        all_poly += info['objects']['polygon']

    img = cv2.imread(img_name)
    thickness = 5
    for it, (obj, poly) in enumerate(zip(all_objects, all_poly)):
        curr_color = colors[color_keys[it % len(color_keys)]]
        pts = np.concatenate([poly['x'][:, None], poly['y'][:, None]], 1)[None,
              :]
        # 이 함수는 16진수 색상 코드를 RGB 값으로 변환하는 함수입니다.
        color = rgb(curr_color[1:])
        img = cv2.polylines(img, pts, True, color, thickness)
    return img


