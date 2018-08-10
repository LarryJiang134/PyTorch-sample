# YOLO v1
> Paper link: [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)

## Unified Detection
- Divide the input into an S x S grid
- Each grid cell predicts B bounding boxes and confidence
- Confidence: reflects how confident the model believe the cell contains an object and how the accurate the box is predicted
    - confidence defined as: ![](http://latex.codecogs.com/gif.latex?\\frac{1}{1+sin(x)})