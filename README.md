# frame_check

### 判断MobileStage多视角相机是否同步（数字板拍摄部分）

依赖包
- python3
- numpy
- opencv-python
- opencv-contrib-python
- pytorch
---

视频文件夹结构：
``` 
<path/to/data>
├── videos
│   ├── 0
│   ├── ...
│   └── 35
├── json
│   ├── 0
│   ├── ...
│   └── 35
└── 
```
抽帧指令，获取视频的帧序列信息，eg：
```
python3 ./utils/extract_oneday_image.py --source <path/to/data> --dest <target_path> --frame
``` 
|选项|名称|备注|
|----|----|----|
|--source|视频文件夹路径|<path/to/data>|
|--dest|保存路径||
|--frame|生成帧序列信息|可选选项，如果后续需要进行同步检测，建议选取，会创建infos目录，并生成视频对应的.info 文件，但对脚本运行时间影响较大|
---

图片文件夹结构：
``` 
<path/to/data>
├── images
│   │── 0
│   │   ├── 000000.jpg
│   │   ├── ...
│   │   ├── 000222.jpg
│   ├── ...
│   └── 35
├── videos
│   ├── 0
│   ├── ...
│   └── 35
├── json
│   ├── 0
│   ├── ...
│   └── 35
├── <infos>
│   ├── <data0.info>
│   ├── ...
│   └── <data35.info>
└── 
```
后续要进行帧同步的检测，必须生成 frames.info 文件，如果抽帧时未选 --frame 参数，运行 frame_info.py 脚本，eg:
```
python3 frame_info.py --src <path/to/data>
# --src : 图片文件夹路径，需要在抽帧后的图片文件夹运行
# 代码执行完成后会在 infos 文件夹下生成 .info 文件，记录时间戳信息
```
---
## 帧序列同步检测
```
python3 camera_check.py --src <path/to/data> --dst <target_path> --pool 3 --model ./finnewmodel.ckpt --bench 0.95 --imgnum 3000 --step 3

```
|选项|名称|备注|
|----|----|----|
|--src|待处理项目文件夹路路径|<path/to/data>|
|--dst|结果文件夹路径|会在路径下生成result文件夹保存结果|
|--pool|开启的进程数|默认为3|
|--model|预测模型的加载路径|finnewmodel.ckpt|
|--imgnum|指定检测的图片数量|如果不给定，则处理所有图片|
|--step|每隔多少帧进行一次检测|视具体的帧情况而定|
|--bench|模型预测的置信度|默认为0.9，建议值在 0.8 - 0.97 之间|


### 脚本执行完成生成结果文件： <br>
---
|名称|备注|
|----|----|
|finset.json|各同步相机的集合信息，所有相机均在同一集合内代表全部同步|
|check.txt|同步过程的中间结果，例如在某一时刻相机同步，相机数字之差>=2|
|result.csv|数字板的预测结果，行代表帧序列，列代表各相机，-1代表预测失败|

---
## 跳帧检测
通过读取 frame.info 里的帧序列时间戳信息获取跳帧情况
```
python3 camera_check.py --src <path/to/data/infos> --dst <target_path> --fps <video_fps>

```
|名称|备注|
|----|----|
|--src|<path/to/data/infos>|
|--dst|会在该文件夹下生成结果文件 check.txt|
|--fps|视频帧率，通常为30或60|

