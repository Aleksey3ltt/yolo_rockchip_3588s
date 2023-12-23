import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
from imutils.video import FPS
import time
from postprocess import yolo_post_process, letterbox_reverse_box
import config
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputtype", required=False, default="cam2",
	help="Select input cam, cam2, file")
args = vars(ap.parse_args())

IMG_SIZE = config.IMG_SIZE
CLASSES = config.CLASSES
DEVICE_COMPATIBLE_NODE = config.DEVICE_COMPATIBLE_NODE
RK3588_RKNN_MODEL = config.RK3588_RKNN_MODEL

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine

    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                    print(os_machine, host)
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host
    
def draw(image, boxes, scores, classes, dw, dh):

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        #print('class: {}, score: {}'.format(CLASSES[cl], score))
        #print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        ##Transform Box            letterbox_reverse_box(x1,   y1,  x2,    y2,     width,            height,            new_width,       new_height,      dw, dh)
        top, left, right, bottom = letterbox_reverse_box(top, left, right, bottom, config.CAM_WIDTH, config.CAM_HEIGHT, config.IMG_SIZE, config.IMG_SIZE, dw, dh)

        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

def letterbox(im, new_shape, color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        print(shape)
    new_unpad=(640,640)
    
    #frame enlargement dw, dh
    dw=5
    dh=5
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, (dw, dh)

def open_cam_usb(dev, width, height):
    if args["inputtype"] == 'cam':
        gst_str = ("uvch264src device=/dev/video{} ! "
               "image/jpeg, width={}, height={}, framerate=30/1 ! "
               "jpegdec ! "
               "video/x-raw, format=BGR ! "
               "appsink").format(dev, width, height)
        vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    elif args["inputtype"] == 'file':
        gst_str = ("filesrc location={} ! "
               "qtdemux name=demux demux. ! queue ! faad ! audioconvert ! audioresample ! autoaudiosink demux. ! "
               "avdec_h264 ! videoscale ! videoconvert ! "
               "appsink").format(args["filename"])		
        vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    elif args["inputtype"] == 'cam2':
        vs = cv2.VideoCapture(dev)
        vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return vs

if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)
    pathModel=os.getcwd()+"/yolo_rockchip_3588s/"    
    print(pathModel+rknn_model)

    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(pathModel+rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    
    print('done')
    print('--> Init runtime environment')

    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print(ret)
    print('done')

    #Create Stream from Webcam
    vs = open_cam_usb(config.CAM_DEV, config.CAM_WIDTH, config.CAM_HEIGHT)

    time.sleep(2.0)
    fps = FPS().start()
    prev_frame_time = 0
    new_frame_time = 0 
 # loop over the frames from the video stream
    while True:

        ret, frame = vs.read()
        if not ret:
            break

 #Show FPS in Pic
        new_frame_time = time.time()
        show_fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        show_fps = int(show_fps)
        show_fps = str("{} FPS".format(show_fps))

        ori_frame = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # Inference
        outputs = rknn_lite.inference(inputs=[frame])

        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

 #YOLO Post process
        boxes, classes, scores = yolo_post_process(input_data)

        img_1 = ori_frame
        if boxes is not None:
            draw(img_1, boxes, scores, classes, dw, dh)   
        # model name
            cv2.putText(img_1, RK3588_RKNN_MODEL, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        #     # show FPS in Frame
            cv2.putText(img_1, show_fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #   show output
            cv2.imshow("yolo result (press 'q' for exit)", img_1)
        else:
            cv2.imshow("yolo result (press 'q' for exit)", ori_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    rknn_lite.release()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()