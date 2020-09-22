# DAVANet

Code repo for the paper "DAVANet: Stereo Deblurring with View Aggregation" (CVPR'19, Oral).&nbsp; [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_DAVANet_Stereo_Deblurring_With_View_Aggregation_CVPR_2019_paper.pdf) &nbsp; [[Project Page]](https://shangchenzhou.com/projects/davanet/)&nbsp;

<p align="center">
  <img width=95% src="https://user-images.githubusercontent.com/14334509/57180102-f9c4b700-6eb7-11e9-927b-42a81ad39d7d.png">
</p>


## Stereo Blur Dataset
<p align="center">
  <img width=100% src="https://user-images.githubusercontent.com/14334509/57179915-e9abd800-6eb5-11e9-86db-2c696fa69bad.png">
</p>

Download the dataset (192.5GB, unzipped 202.2GB) from [[Data Website]](https://stereoblur.shangchenzhou.com/).

## Pretrained Models

You could download the pretrained model (34.8MB) of DAVANet from [[Here]](https://drive.google.com/file/d/1oVhKnPe_zrRa_JQUinW52ycJ2EGoAcHG/view?usp=sharing). 

(Note that the model does not need to unzip, just load it directly.)

## Prerequisites

- Linux (tested on Ubuntu 14.04/16.04)
- Python 2.7+
- Pytorch 0.4.1
- easydict
- tensorboardX
- pyexr

#### Installation

```
pip install -r requirements.txt
```

## Get Started

Use the following command to train the neural network:

```
python runner.py 
        --phase 'train'\
        --data [dataset path]\
        --out [output path]
```

Use the following command to test the neural network:

```
python runner.py \
        --phase 'test'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
Use the following command to resume training the neural network:

```
python runner.py 
        --phase 'resume'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
You can also use the following simple command, with changing the settings in config.py:

```
python runner.py
```

## Results on the testing dataset

<p align="center">
  <img width=100% src="https://user-images.githubusercontent.com/14334509/57179916-ea446e80-6eb5-11e9-8eb6-98fb878810e7.png">
</p>

## Citation
If you find DAVANet, or Stereo Blur Dataset useful in your research, please consider citing:

```
@inproceedings{zhou2019davanet,
  title={{DAVANet}: Stereo Deblurring with View Aggregation},
  author={Zhou, Shangchen and Zhang, Jiawei and Zuo, Wangmeng and Xie, Haozhe and Pan, Jinshan and Ren, Jimmy},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to shangchenzhou@gmail.com

## Reference
[1] Zhe Hu, Li Xu, and Ming-Hsuan Yang. Joint depth estimation and camera shake removal from single blurry image. In *CVPR*, 2014.

[2] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In *CVPR*, 2017.

[3] Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, and Jiri Matas. Deblurgan: Blind motion deblurring using conditional adversarial networks. In CVPR, 2018.

[4] Jiawei Zhang, Jinshan Pan, Jimmy Ren, Yibing Song, Lin- chao Bao, Rynson WH Lau, and Ming-Hsuan Yang. Dynamic scene deblurring using spatially variant recurrent neural networks. In *CVPR*, 2018. 

[5] Xin Tao, Hongyun Gao, Xiaoyong Shen, Jue Wang, and Jiaya Jia. Scale-recurrent network for deep image deblurring. In *CVPR*, 2018.

## License

This project is open sourced under MIT license.
