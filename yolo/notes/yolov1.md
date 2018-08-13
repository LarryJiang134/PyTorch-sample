# YOLO v1
> Paper link: [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)

## Unified Detection
- Divide the input into an S x S grid
- Each grid cell predicts B bounding boxes and confidence
    - Note: grid cell != bouding box
- Confidence: reflects how confident the model believe the cell contains an object and how the accurate the box is predicted
    - confidence defined as: <img src="http://latex.codecogs.com/gif.latex?Pr(Object)&space;*&space;IOU^{truth}_{pred}" title="Pr(Object) * IOU^{truth}_{pred}" />
    - confidence = 0, if no object in the cell
- Each bounding box consists of 5 predictions: `x, y, w, h, confidence`
    - x, y: center of the box
    - w, h: relative to the whole image
    - confidence: the confidence prediction represents the IOU between the predicted box and any ground truth box
- Each grid cell predicts C conditional class probabilities: <img src="http://latex.codecogs.com/gif.latex?Pr(Class_i|Object)" title="Pr(Class_i|Object)" />
    - predict 1 set of class probabilities per grid cell, regardless of the number of bounding boxes B
    - At test time, we get class-specific confidence score for each box by computing:
    
    <img src="http://latex.codecogs.com/gif.latex?Pr(Class_i|Object)&space;*&space;Pr(Object)&space;*&space;IOU^{truth}_{pred}&space;=&space;Pr(Class_i)&space;*&space;IOU^{truth}_{pred}\tag{1}" title="Pr(Class_i|Object) * Pr(Object) * IOU^{truth}_{pred} = Pr(Class_i) * IOU^{truth}_{pred}\tag{1}" />
    
    ![yolov1-1](_image/yolov1-1.png)
- On `PASCAL VOC`, set S=7, B=2, C=20 (20 classes), hence tensor number is 7x7x30
    
### Network Design
- Initial `conv` layers extract features, `fc` layers predict the output probabilities and coordinates
- 24 `conv` layers + 2 `fc` layers
- `Dropout` layer with 0.5 droping rate after `fc-1`

![yolov1-2](_image/yolov1-2.png)

### Training
- Pretrain `conv` layers on the `ImageNet`
    - Used first 20 `conv` layers followed by 1 `avgPool` layer and 1 `fc` layer
    - Training time: 1 week, top-5 accuracy: 88% on `val` set
- Training by adding 4 `conv` layers and 2 `fc` layers with random initialization
    - Input resolution set to 448 x 448
    - Normalize the bounding box `w` and `h` by the image width and height (0~1)
    - Making `x` and `y` to be offsets of a particular grid cell location (0~1)
    - Activation function: <img src="http://latex.codecogs.com/gif.latex?\phi(x)&space;=&space;\left\{&space;\begin{array}{ll}&space;x,&space;if&space;x>0\\&space;\\&space;0.1x,&space;otherwise&space;\end{array}&space;\right." title="\phi(x) = \left\{ \begin{array}{ll} x, if x>0\\ \\ 0.1x, otherwise \end{array} \right." />
    - Optimize for `sum-square error` (easy to optimize but isn't perfectly suitable for the task)
    - To make `sum-square error` work better
        - Increase the loss of bounding box coordinate prediction
        - Decrease the loss of confidence for boxes don't contain objects
        - <img src="http://latex.codecogs.com/gif.latex?\lambda_coord&space;=&space;5" title="\lambda_coord = 5" /> and <img src="http://latex.codecogs.com/gif.latex?\lambda_noobj&space;=&space;0.5" title="\lambda_noobj = 0.5" />
        - Predict the square root of the bounding box width and height to make smaller boxes matter more
    - Loss function:
        - <img src="http://latex.codecogs.com/gif.latex?\lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}[(x_i&space;-&space;\^{x}_i)^2&space;&plus;&space;(y_i&space;-&space;\^{y}_i)^2]\\&space;&plus;&space;\lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}[(\sqrt{w_i}&space;-&space;\sqrt{\^{w}_i})^2&space;&plus;&space;(\sqrt{h_i}&space;-&space;\sqrt{\^{h}_i})^2]&space;&plus;&space;\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}(C_i&space;-&space;\^{C}_i)^2\\&space;&plus;&space;\lambda_{noobj}\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}(C_i&space;-&space;\^{C}_i)^2&space;&plus;&space;\sum^{S^2}_{i=0}\mathbb{I}^{obj}_{i}\sum_{c&space;\in&space;classes}(p_i(c)&space;-&space;\^{p}_i(c))^2\\" title="\lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}[(x_i - \^{x}_i)^2 + (y_i - \^{y}_i)^2]\\ + \lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}[(\sqrt{w_i} - \sqrt{\^{w}_i})^2 + (\sqrt{h_i} - \sqrt{\^{h}_i})^2] + \sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}(C_i - \^{C}_i)^2\\ + \lambda_{noobj}\sum^{S^2}_{i=0}\sum^{B}_{j=0}\mathbb{I}^{obj}_{ij}(C_i - \^{C}_i)^2 + \sum^{S^2}_{i=0}\mathbb{I}^{obj}_{i}\sum_{c \in classes}(p_i(c) - \^{p}_i(c))^2\\" />
        - <img src="http://latex.codecogs.com/gif.latex?\mathbb{I}^{obj}_{ij}" title="\mathbb{I}^{obj}_{ij}" /> denotes if object appears in j-th bounding box of cell i
    - In the paper, the training setting is:
        - `training set`: PASCAL VOC 2007 and 2012
        - `epoches`: 135
        - `batch size`: 64
        - `momentum`: 0.9
        - `decay`: 0.0005
        - `learning rate`:
            - 1-st epoch: from 0.001 to 0.01 (to avoid possible diverge due to initialization)
            - 2-nd ~ 75-th epoch: 0.01
            - 76-th ~ 105-th epoch: 0.001
            - 106-th ~ 135-th epoch: 0.0001
        - `data augmentation`:
            - random scaling
            - translations of up to 20% of original image size
            - randomly adjust the exposure and saturation of the image (up to the factor of 1.5 in HSV color space)

### Limitation
- Not suitable for dense objects detection
- Struggles with small objects that appears in groups

