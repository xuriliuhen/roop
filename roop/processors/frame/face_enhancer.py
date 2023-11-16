from typing import Any, List, Callable
import cv2
import threading
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

import insightface.utils.ImageProcessor as ImageProcessor

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    source_face = temp_face.copy()
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        
        import numpy as np
        import numexpr as ne
        mask = np.zeros((temp_face.shape[0], temp_face.shape[1], 1), dtype=np.float32)
        poly = list()
        lmks = target_face['landmark_2d_106']
        pt_boundarys = cv2.convexHull(lmks, clockwise=True, returnPoints=True)
        '''temp_face_show = temp_face.copy()
        pt_mean = [0, 0]
        # print("pt-boundarys--->", pt_boundarys)
        pt_lists = list()
        for pt in pt_boundarys:
            # print("pt---", pt)
            pt_lists.append([pt[0][0], pt[0][1]])
            pt_mean = [pt_mean[0] + pt[0][0], pt_mean[1] + pt[0][1]]
        
        # print("pt_lists --- >", pt_lists)
        # print("pt_mean -- >", pt_mean)
        # print("pt_boundarys.shape -- >", pt_boundarys.shape)
        pt_mean = [pt_mean[0]/pt_boundarys.shape[0], pt_mean[1]/pt_boundarys.shape[0]]
        pt_mean = np.dot(pt_mean, M.T[0:2]) + M.T[2]'''

        pt_mean = [0, 0]
        pt_lists = list()
        for pt in pt_boundarys:
            pt_lists.append([pt[0][0], pt[0][1]])
            pt_mean = [pt_mean[0] + pt[0][0], pt_mean[1] + pt[0][1]]
        pt_mean = [pt_mean[0]/pt_boundarys.shape[0], pt_mean[1]/pt_boundarys.shape[0]]

        for pt in pt_boundarys:
            # if pt[0][0] < pt_mean[0]:
            #     pt[0][0] = pt_mean[0] - (pt_mean[0]-pt[0][0])*1.01
            # if pt[0][0] > pt_mean[0]:
            #     pt[0][0] = (pt[0][0]-pt_mean[0])*1.1 + pt_mean[0]

            pt = [pt[0][0] - start_x, pt[0][1] - start_y]
            poly.append([int(pt[0]), int(pt[1])])
        poly = np.array(poly, np.int32)
        cv2.fillPoly(mask, [poly], (1.0,1.0,1.0))

        # mask = mask*255.0
        # mask_show = mask*255
        # cv2.imshow("mask_show", mask_show)
        # cv2.imshow("swapped", temp_face)
        # cv2.imshow("source", source_face)
        # cv2.imwrite("mask_show.jpg", mask_show)
        # cv2.imwrite("swapped.jpg", temp_face)
        # cv2.imwrite("source.jpg", source_face)
        # cv2.waitKey(0)
        
        frame_mask = ImageProcessor(mask).erode_blur(5,5,fade_to_border=True).get_image('HWC')
        # cv2.imwrite("mask.jpg", frame_mask)
        one_f = np.float32(1.0)
        bgr_copy = temp_face.copy()
        swapped_face = ImageProcessor(bgr_copy).to_ufloat32().get_image('HWC')
        source_face = ImageProcessor(source_face).to_ufloat32().get_image('HWC')
        res_swapped = ne.evaluate('source_face*(one_f-frame_mask)+swapped_face*frame_mask')
        bgr_fake = ImageProcessor(res_swapped, copy=True).to_uint8().get_image('HWC')
        temp_frame[start_y:end_y, start_x:end_x] = bgr_fake
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)

        source_image = cv2.imread(temp_frame_path, flags=cv2.IMREAD_UNCHANGED)
        # print("enhance_source_image.shape -- >", source_image.shape)
        if source_image.shape[2] == 4:
            # print("run enhance face ----- ")
            result_tmp = source_image.copy()
            result_tmp[...,0:3] = result
            result = result_tmp.copy()

        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)

    
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
