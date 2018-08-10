# YOLO v1
> Paper link: [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)

## Unified Detection
- Divide the input into an S x S grid
- Each grid cell predicts B bounding boxes and confidence
    - Note: grid cell != bouding box
- Confidence: reflects how confident the model believe the cell contains an object and how the accurate the box is predicted
    - confidence defined as: <img src="http://latex.codecogs.com/gif.latex?Pr(Object)&space;*&space;IOU^{truth}_{pred}" title="Pr(Object) * IOU^{truth}_{pred}" />
    - confidence = 0, if no object in the cell
- Each bounding box consists of 5 predictions: x, y, w, h, confidence
    - x, y: center of the box
    - w, h: relative to the whole image
    - confidence: the confidence prediction represents the IOU between the predicted box and any ground truth box
- Each grid cell predicts C conditional class probabilities: <img src="http://latex.codecogs.com/gif.latex?Pr(Class_i|Object)" title="Pr(Class_i|Object)" />
    - predict 1 set of class probabilities per grid cell, regardless of the number of bounding boxes B
    - At test time: <img src="http://latex.codecogs.com/gif.latex?Pr(Class_i|Object)&space;*&space;Pr(Object)&space;*&space;IOU^{truth}_{pred}&space;=&space;Pr(Class_i)&space;*&space;IOU^{truth}_{pred}\tag{1}" title="Pr(Class_i|Object) * Pr(Object) * IOU^{truth}_{pred} = Pr(Class_i) * IOU^{truth}_{pred}\tag{1}" />
    