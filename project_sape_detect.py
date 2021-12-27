#!/usr/bin/env python
# coding: utf-8
# %%


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
import shutil
import cv2
from glob import glob

import torch
import torch.backends.cudnn as cudnn
from gtts import gTTS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from time import time, sleep

@torch.no_grad()
def run_people(weights='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/crowdhuman_yolov5m.pt',  # model.pt path(s)
        source='C:/Users/Playdata/PycharmProjects/404prjtest/team404/output_image/output.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/yolov5/runs/detect/',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    global people
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = cfheck_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            np_det = det.numpy()
            print('사람 몇 명?', np_det.shape[0])
            people = np_det.shape[0]
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            for ind, npd in enumerate(np_det):
                x11, y11, x22, y22, accuraa, classss = npd 
                print('좌표입니다.')
                print(x11, y11, x22, y22)
                imgggggg = cv2.imread('C:/Users/Playdata/PycharmProjects/404prjtest/team404/output_image/output.jpg')
                print('원본 read', imgggggg.shape)
                x11 = int(x11) # / 640 * im0s.copy().shape[0] / 640 8 640 640 480
                x22 = int(x22) # / 640 * im0s.copy().shape[1])
                y11 = int(y11) # / 480 * im0s.copy().shape[0])
                y22 = int(y22) # / 480 * im0s.copy().shape[0])
                if x11 < 10:
                    x11 = 10
                if x22 > im0s.copy().shape[0] - 10:
                    x22 = int(im0s.copy().shape[0] - 11)
                if y11 < 10:
                    y11 = 10
                if y22 > im0s.copy().shape[1] - 10:
                    y22 = int(im0s.copy().shape[1] - 11)
                print(im0.shape)
                #cv2.imwrite('/root/hardhat/content/yolov5/im0s.jpg', im0s)
                file_path = "C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/images/"+ str(ind) +".jpg"
                #file_path2 = "C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/images/"+ str(ind) +"2.jpg"
                img = cv2.imread(file_path) 
                print(img)
                print(file_path)
                cv2.imwrite(file_path, im0s.copy()[y11-10:y22+10,x11-10:x22+10])
                #cv2.imwrite(file_path2, imgggggg.copy()[y11-10:y22+10,x11-10:x22+10])
                #cv2.imwrite(file_path, im0s)
                
               # cv2.imwrite("C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/images/"+ str(ind) +".jpg", im0s.copy()[y11-10:y22+10,x11-10:x22+10])
                #cv2.imwrite('./3.jpg', im0s.copy()[y11-10:y22+10,x11-10:x22+10])
                #C:\Users\Playdata\PycharmProjects\404prjtest\team404\hardhat\content\yolov5\images
                print('통관')
        #return np_det.shape[0]

@torch.no_grad()
def run_equip(weights='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/20211218/best (1).pt',  # model.pt path(s)
        source='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/images/',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    global hardhat, vest
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        print(im0s.copy().shape)

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #print('det는', det)
            np_det = det.numpy()
            print('np_det는', np_det)
            
            answers_hardhat = [hardhat.append(int(i[-1])) for i in np_det if int(i[-1]) == 0] #결과 추가
            answers_vest = [vest.append(int(i[-1])) for i in np_det if int(i[-1]) == 1] #결과 추가
            
            #print('\n\n\n--------------------\n', answers_hardhat, answers_vest, '\n\n\n--------------------\n')
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    #print('\n\n\n--------------------\n', answers_hardhat, answers_vest)
    #return answers_hardhat, answers_vest

def equip_parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/20211218/last (1).pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/yolov5/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def peoeple_parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/crowdhuman_yolov5m.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'C:/Users/Playdata/PycharmProjects/404prjtest/team404/output_image/output.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/yolov5/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main_equip(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run_equip(**vars(opt))

def main_people(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run_people(**vars(opt))
#def tts():
    
# def traffic_sign():
#     try:
#         print(f'사람 수는 : {len(answers)}, 각 클래스 별 answer는 : {answers}') 
#     except:
#         ser.close()
def reading_from_user(input_text, index):
    language = 'ko'
    slow_audio_speed = False
    filename = f"C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/mp3_file/output_{index}.mp3"
    audio_created =gTTS(text=input_text, lang=language, slow=slow_audio_speed)
    audio_created.save(filename)
    os.system(f'start {filename}') #실행까지 #윈도우 
    #os.system(f'afplay {filename}') #리눅스
if __name__ == "__main__":
    start = time()
    re_tts = None
    while True:
        
        now = time()
        if now - start > 5:
            start = now
        else:
            sleep(0.3)
            continue
        #ser = serial.Serial('/dev/cu.usbmodem1101', 9600)
        #traffic_sign(ser, '2')
        people_opt = peoeple_parse_opt()
        people = 0
        file_path = 'C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/content/yolov5/images/'

        if os.path.exists(file_path):
            shutil.rmtree(file_path)
            os.makedirs(file_path)
            print(1)
        else:
            os.makedirs(file_path)
        main_people(people_opt)
        if people < 1:
            now_answer = '작업자가 없습니다.'
            if re_tts == now_answer:
                continue
            else:
                re_tts = now_answer
            reading_from_user(now_answer, '0')
            print('작업자가 없습니다.')
            open('C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/detect_text.txt', 'w').write('people 0; hardhat 0; vest 0;')
            continue
        equip_opt = equip_parse_opt()
        hardhat, vest = [], []
        main_equip(equip_opt)
        hardhat = len(hardhat)
        vest = len(vest)
 
        print('  작업자 명수는 :', people)
        print('    박 명  수는 :', '유재석 친구이자 2인자')
        print('  안전모 개수는 :', hardhat)
        print('안전조끼 개수는 :', vest)
        answer = int(people * 2 - hardhat - vest)
        if answer < 1:
            now_answer = f'작업자 인원 총 {people}명이며 현재 안전한 상태입니다.'
        elif answer < 2:
            now_answer = f'작업자 인원 총 {people}명이며 주의하셔야 합니다.'
        else:
            now_answer = f'작업자 인원 총 {people}명이며 위험합니다! 안전 장비를 착용하십시오'
            
        if re_tts == now_answer:
            continue
        else:
            re_tts = now_answer
        reading_from_user(now_answer, '0')

        content = f'people {people};hardhat {hardhat};vest {vest};'
        open('C:/Users/Playdata/PycharmProjects/404prjtest/team404/hardhat/detect_text.txt', 'w').write(content)

# %%
O
