"""
yolo detection demo
"""
import argparse
import cv2

from yolo.darknet import Darknet
from yolo.utils import *


def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='PyTorch YOLOv2 LOGO detection Training')
    parser.add_argument('--data', '-d', type=str, default='data/logo.data', help='data definition file')
    parser.add_argument('--video', '-v', type=str, default='none', help='testing video, none if using webcam')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolo_v2.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w', type=str, default='weights/pre-trained/yolov2.weights', help='initial weights')

    return parser.parse_args()


def main():
    m = Darknet(args.config)
    m.print_network()
    m.load_weights(args.weights)
    print('Loading weights from %s... Done!' % (args.weights))

    if m.num_classes == 20:
        namesfile = 'weights/voc.names'
    elif m.num_classes == 80:
        namesfile = 'weights/coco.names'
    elif m.num_classes == 1:
        namesfile = 'weights/logo.names'
    else:
        namesfile = 'data/names'
    print("{} is used for classification".format(namesfile))
    class_names = load_class_names(namesfile)

    use_cuda = True
    if use_cuda:
        m.cuda()

    if args.video != 'none':
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.imshow(args.config, draw_img)
            cv2.waitKey(1)
        else:
            print("Unable to read image")
            exit(-1)


if __name__ == '__main__':
    args = arg_parse()
    main()