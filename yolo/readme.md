# PyTorch Implementation of YOLO (v2)
Note, implementation not yet done ! ! !  


[paper notes](/notes)
## TODO list
- [ ] darknet
- [ ] detect
- [ ] demo
- [ ] train
- [ ] paper notes 

## Setup
From [link for YOLOv2](https://pjreddie.com/darknet/yolov2/)  
- download weight files into `./weights` directory  
- download cfg files into `./cfg` directory

## Run
Test on webcam:
```
python demo.py --weights 'weights_dir'
```

Test on video:
```
python demo.py --video 'video_dir' --weights 'weights_dir'
```

## Reference:  
> - YOLO(v1): [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)
> - YOLO(v2): 
>   - [YOLO: Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
>   - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
> - YOLO(v3): [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
