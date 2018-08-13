# YOLO v3
> Paper link: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

> link: [How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/tag/series-yolo/)

## A Fully Convolutional Neural Network
- YOLO is a `Fully Connected Neural Network`
- YOLO uses only `conv` layers (has 75 layers with skip connections and upsampling layers)
- downsample use `conv` with `stride` 2
- YOLO invariant to input size but better use constant size

## Interpreting the output
- Predictions are made through a 1x1 `convolution`
- output is a feature map
- `(B x (5 + C))` entries in the feature map
  - B: number of the bounding boxes each neuron can predict
  - each bounding box has 5 + C atrributes, describing the center coordinates, the dimensions, the objectness score, and the C class confidences for each bounding box
- In YOLO v3, each neuron predicts 3 bounding boxes
- You expect each cell of the feature map to predict an object through one of it's bounding boxes if the center of the object falls in the receptive field of that cell

![yolo-5](./_image/yolov3-1.png)

## Anchor Boxes
- Has three `anchors`

## Making Predictions
- Formulas for obtaining bounding box Predictions
  - <img src="http://latex.codecogs.com/gif.latex?$b_x&space;=&space;\sigma(t_x)&plus;c_x$" title="$b_x = \sigma(t_x)+c_x$" />
  - <img src="http://latex.codecogs.com/gif.latex?$b_y&space;=&space;\sigma(t_y)&plus;c_y$" title="$b_y = \sigma(t_y)+c_y$" />
  - <img src="http://latex.codecogs.com/gif.latex?$b_w&space;=&space;p_we^{t_w}$" title="$b_w = p_we^{t_w}$" />
  - <img src="http://latex.codecogs.com/gif.latex?$b_h&space;=&space;p_he^{t_h}$" title="$b_h = p_he^{t_h}$" />

- `Center coordinates` go through sigmoid function to force the value between 0-1
  - this makes sure that the center lies in the current cell box (neuron)

- `Dimensions of the Bounding Box` predicted by applying a `log-space transform` to the output and then multiplying with an anchor
  - <img src="http://latex.codecogs.com/gif.latex?$b_w$,&space;$b_h$" title="$b_w$, $b_h$" /> are normalized by images width and height

![yolo-regression-1](./_image/yolov3-2.png)

- `Objectness Score` represents the probability that an object (whatever it is) is contained inside a bounding box

- `Class Confidences` represent the probabilities of the detected object belonging to a particular class

## Prediction across different scales
- `YOLO v3` uses 3 `scales` (stride 32, 16, 8)
- Downsample the input until the first Detection layer
- Layers are upsampled (by factor 2) and concatenated with feature maps of a previous layers having identical feature map sizes
- Process repeated and the final prediction are made at the layer with stride 8
- Each scale, each cell predicts 3 bounding boxes using 3 anchors

![yolo_Scales-1](./_image/yolov3-3.png)

## Output Processing
*e.g*. For an image of size 416 x 416, YOLO predicts ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10647 bounding boxes. But need to reduce 10647 to 1 as the final Prediction
- `Thresholding by Object Confidence`(boxes having objectness scores below a threshold are ignored)
- `Non-maximum Suppression` prevents multiple detections of the same object
