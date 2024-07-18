# View Morphing

view morphing algorithm implementation of [sigg96] by Steven M. Seitz.

http://www.cs.cmu.edu/~seitz/papers/sigg96.pdf



### Clone Source
```bash
https://github.com/zongwave/viewmorphing.git
```

### Install Dependency
[OpenCV](https://github.com/opencv/opencv), [libXCam](https://github.com/intel/libxcam)



### Compile Source Code
```bash
cd viewmorphing && make
```


### Run Application
```bash
./run_freeview.sh minion_1.jpg minion_2.jpg 8
```

#### Usage
        --input          input image(JPEG/PNG/BMP/YUV/NV12)
        --output         output image(JPEG/PNG/BMP/YUV/NV12/MP4)
        --save           save output images
        --interp-count   optional, interpolation frame count: 2
        --in-w           optional, input width, default: 1280
        --in-h           optional, input height, default: 800
        --out-w          optional, output width, default: 1920
        --out-h          optional, output height, default: 640
        --continuous     optional, continuous output image(YUV/NV12/MP4)


## Input
![left image](https://github.com/zongwave/viewmorphing/blob/main/images/minion_1.jpg)
![right image](https://github.com/zongwave/viewmorphing/blob/main/images/minion_2.jpg)

## Processing
![View Morphing](https://github.com/zongwave/viewmorphing/blob/main/images/free_view_projection.png)


## Output
![right image](https://github.com/zongwave/viewmorphing/blob/main/images/minion_out.gif)
