import sys
import argparse
import numpy as np
import cv2
import time
from PIL import Image
from time import sleep
import multiprocessing as mp
from edgetpu.basic import edgetpu_utils
from pose_engine import PoseEngine
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.detection.engine import DetectionEngine

pose_lastresults = None
deep_lastresults = None
ssd_lastresults = None
processes = []
pose_frameBuffer = None
deep_frameBuffer = None
ssd_frameBuffer = None
pose_results = None
deep_results = None
ssd_results = None
fps = ""
pose_detectfps = ""
deep_detectfps = ""
ssd_detectfps = ""
framecount = 0
pose_detectframecount = 0
deep_detectframecount = 0
ssd_detectframecount = 0
time1 = 0
time2 = 0

box_color = (255, 128, 0)
box_thickness = 1
label_background_color = (125, 175, 75)
label_text_color = (255, 255, 255)
percentage = 0.0

# COCO Labels
SSD_LABELS = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','',
              'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep',
              'cow','elephant','bear','','zebra','giraffe','backpack','umbrella','','',
              'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
              'skateboard','surfboard','tennis racket','bottle','','wine glass','cup','fork','knife','spoon',
              'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
              'cake','chair','couch','potted plant','bed','','dining table','','','toilet',
              '','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
              'sink','refrigerator','','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

# Deeplab color palettes
DEEPLAB_PALETTE = Image.open("models/colorpalette.png").getpalette()

# Posenet Edges
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


def camThread(pose_results, deep_results, ssd_results,
              pose_frameBuffer, deep_frameBuffer, ssd_frameBuffer,
              camera_width, camera_height, vidfps, usbcamno, videofile):

    global fps
    global pose_detectfps
    global deep_detectfps
    global ssd_detectfps
    global framecount
    global pose_detectframecount
    global deep_detectframecount
    global ssd_detectframecount
    global time1
    global time2
    global pose_lastresults
    global deep_lastresults
    global ssd_lastresults
    global cam
    global window_name
    global waittime

    if videofile == "":
        cam = cv2.VideoCapture(usbcamno)
        cam.set(cv2.CAP_PROP_FPS, vidfps)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        waittime = 1
        window_name = "USB Camera"
    else:
        cam = cv2.VideoCapture(videofile)
        waittime = vidfps
        window_name = "Movie File"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        ret, color_image = cam.read()
        if not ret:
            continue

        if pose_frameBuffer.full():
            pose_frameBuffer.get()
        if deep_frameBuffer.full():
            deep_frameBuffer.get()
        if ssd_frameBuffer.full():
            ssd_frameBuffer.get()

        frames = cv2.resize(color_image, (camera_width, camera_height)).copy()
        pose_frameBuffer.put(cv2.resize(color_image, (640, 480)).copy())
        deep_frameBuffer.put(cv2.resize(color_image, (513, 513)).copy())
        ssd_frameBuffer.put(cv2.resize(color_image, (640, 480)).copy())

        res = None

        # Posenet
        if not pose_results.empty():
            res = pose_results.get(False)
            pose_detectframecount += 1
            imdraw = pose_overlay_on_image(frames, res)
            pose_lastresults = res
        else:
            imdraw = pose_overlay_on_image(frames, pose_lastresults)

        # MobileNet-SSD
        if not ssd_results.empty():
            res = ssd_results.get(False)
            ssd_detectframecount += 1
            imdraw = ssd_overlay_on_image(imdraw, res)
            ssd_lastresults = res
        else:
            imdraw = ssd_overlay_on_image(imdraw, ssd_lastresults)

        # Deeplabv3
        if not deep_results.empty():
            res = deep_results.get(False)
            deep_detectframecount += 1
            imdraw = deep_overlay_on_image(imdraw, res, camera_width, camera_height)
            deep_lastresults = res
        else:
            imdraw = deep_overlay_on_image(imdraw, deep_lastresults, camera_width, camera_height)

        cv2.putText(imdraw, fps,            (camera_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(imdraw, pose_detectfps, (camera_width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(imdraw, deep_detectfps, (camera_width-170,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(imdraw, ssd_detectfps,  (camera_width-170,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, imdraw)

        if cv2.waitKey(waittime)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        # Posenet
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            pose_detectfps = "(Posenet)  {:.1f} FPS".format(pose_detectframecount/time2)
            deep_detectfps = "(Deeplab)  {:.1f} FPS".format(deep_detectframecount/time2)
            ssd_detectfps  = "(SSD)      {:.1f} FPS".format(ssd_detectframecount/time2)
            framecount = 0
            pose_detectframecount = 0
            deep_detectframecount = 0
            ssd_detectframecount  = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


def pose_inferencer(results, frameBuffer, model, device):

    pose_engine = None
    pose_engine = PoseEngine(model, device)
    print("Loaded Graphs!!! (Posenet)")

    while True:

        if frameBuffer.empty():
            continue

        # Run inference.
        color_image = frameBuffer.get()
        prepimg_pose = color_image[:, :, ::-1].copy()
        tinf = time.perf_counter()
        result_pose, inference_time = pose_engine.DetectPosesInImage(prepimg_pose)
        print(time.perf_counter() - tinf, "sec (Posenet)")
        results.put(result_pose)


def deep_inferencer(results, frameBuffer, model, device):

    deep_engine = None
    deep_engine = BasicEngine(model, device)
    print("Loaded Graphs!!! (Deeplab)")

    while True:

        if frameBuffer.empty():
            continue

        # Run inference.
        color_image = frameBuffer.get()
        prepimg_deep = color_image[:, :, ::-1].copy()
        prepimg_deep = prepimg_deep.flatten()
        tinf = time.perf_counter()
        latency, result_deep = deep_engine.run_inference(prepimg_deep)
        print(time.perf_counter() - tinf, "sec (Deeplab)")
        results.put(result_deep)


def ssd_inferencer(results, frameBuffer, model, device):

    ssd_engine  = None
    ssd_engine = DetectionEngine(model, device)
    print("Loaded Graphs!!! (SSD)")

    while True:

        if frameBuffer.empty():
            continue

        # Run inference.
        color_image = frameBuffer.get()
        prepimg_ssd = color_image[:, :, ::-1].copy()
        prepimg_ssd = Image.fromarray(prepimg_ssd)
        tinf = time.perf_counter()
        result_ssd = ssd_engine.detect_with_image(prepimg_ssd, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=10)
        print(time.perf_counter() - tinf, "sec (SSD)")
        results.put(result_ssd)


def draw_pose(img, pose, threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        img = cv2.circle(img, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5, (0, 255, 0), -1)

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        img = cv2.line(img, (ax, ay), (bx, by), (0, 255, 255), 2)


def pose_overlay_on_image(frames, result):

    color_image = frames

    if isinstance(result, type(None)):
        return color_image
    img_cp = color_image.copy()

    for pose in result:
        draw_pose(img_cp, pose)

    return img_cp


def deep_overlay_on_image(frames, result, width, height):

    color_image = frames

    if isinstance(result, type(None)):
        return color_image
    img_cp = color_image.copy()

    outputimg = np.reshape(np.uint8(result), (513, 513))
    outputimg = cv2.resize(outputimg, (width, height))
    outputimg = Image.fromarray(outputimg, mode="P")
    outputimg.putpalette(DEEPLAB_PALETTE)
    outputimg = outputimg.convert("RGB")
    outputimg = np.asarray(outputimg)
    outputimg = cv2.cvtColor(outputimg, cv2.COLOR_RGB2BGR)
    img_cp = cv2.addWeighted(img_cp, 1.0, outputimg, 0.9, 0)

    return img_cp


def ssd_overlay_on_image(frames, result):

    color_image = frames

    if isinstance(result, type(None)):
        return color_image
    img_cp = color_image.copy()

    for obj in result:
        box = obj.bounding_box.flatten().tolist()
        box_left = int(box[0])
        box_top = int(box[1])
        box_right = int(box[2])
        box_bottom = int(box[3])
        cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        percentage = int(obj.score * 100)
        label_text = SSD_LABELS[obj.label_id] + " (" + str(percentage) + "%)" 

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
        cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    return img_cp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_model", default="models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite", help="Path of the posenet model.")
    parser.add_argument("--deep_model", default="models/deeplabv3_mnv2_dm05_pascal_trainaug_edgetpu.tflite", help="Path of the deeplabv3 model.")
    parser.add_argument("--ssd_model", default="models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", help="Path of the mobilenet-ssd model.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    parser.add_argument('--videofile', default="", help='Path to input video file. (Default="")')
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')
    parser.add_argument('--camera_width', type=int, default=640, help='USB Camera resolution (width). (Default=640)')
    parser.add_argument('--camera_height', type=int, default=480, help='USB Camera resolution (height). (Default=480)')
    args = parser.parse_args()

    pose_model    = args.pose_model
    deep_model    = args.deep_model
    ssd_model     = args.ssd_model
    usbcamno      = args.usbcamno
    vidfps        = args.vidfps
    videofile     = args.videofile
    camera_width  = args.camera_width
    camera_height = args.camera_height

    try:
        mp.set_start_method('forkserver')
        pose_frameBuffer = mp.Queue(10)
        deep_frameBuffer = mp.Queue(10)
        ssd_frameBuffer  = mp.Queue(10)
        pose_results = mp.Queue()
        deep_results = mp.Queue()
        ssd_results  = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(pose_results, deep_results, ssd_results,
                             pose_frameBuffer, deep_frameBuffer, ssd_frameBuffer,
                             camera_width, camera_height, vidfps, usbcamno, videofile),
                       daemon=True)
        p.start()
        processes.append(p)

        # Activation of inferencer
        devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
        print(devices)

        # Posenet
        if len(devices) >= 1:
            p = mp.Process(target=pose_inferencer,
                           args=(pose_results, pose_frameBuffer, pose_model, devices[0]),
                           daemon=True)
            p.start()
            processes.append(p)

        # DeeplabV3
        if len(devices) >= 2:
            p = mp.Process(target=ssd_inferencer,
                           args=(ssd_results, ssd_frameBuffer, ssd_model, devices[1]),
                           daemon=True)
            p.start()
            processes.append(p)

        # MobileNet-SSD v2
        if len(devices) >= 3:
            p = mp.Process(target=deep_inferencer,
                           args=(deep_results, deep_frameBuffer, deep_model, devices[2]),
                           daemon=True)
            p.start()
            processes.append(p)

        while True:
            sleep(1)

    finally:
        for p in range(len(processes)):
            processes[p].terminate()
