# Multi-rate Video Compressive Sensing for Fixed Scene Measurement

Official code for paper "Multi-rate Video Compressive Sensing for Fixed Scene Measurement" at International Conference on Video and Image Processing 2021.

(Also as a part for chapter 5 of my Ph.D. thesis)

## Platform Setup

Before running the code, ubuntu (above or equal to 20.04) and python (>= 3.9) is required, or alternatively windows 10 or 11.

You should install pytorch and torchvision by yourself. (pytorch >= 1.9.0)

This code has been tested on a computer with 2 Nvidia RTX 3090 GPUs.

## Training

For different measurement rate, simply run:

- train_LR.py

- train_HR.py

## Testing

- video_test.py

## Citing this work

Please cite our paper in your publications if it helps your research:

```bib
@article{du2021multi,
    author = {Du, Jiang and Xie, Xuemei and Shi, Guangming},
    title = {Multi-rate Video Compressive Sensing for Fixed Scene Measurement},
    journal={International Conference on Video and Image Processing},
    year = {2021},
    month = {December}
}
```

## LICENSE

MIT License
