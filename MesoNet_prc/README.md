# MesoNet

You can find here the implementation of the network architecture and the dataset used in our paper on digital forensics. It was accepted at the [WIFS 2018 conference](http://wifs2018.comp.polyu.edu.hk).

[Link to full paper](https://arxiv.org/abs/1809.00888)

## Requirements

- Python 3.7
- Numpy 1.14.2
- Keras 2.1.5

If you want to use the complete pipeline with the face extraction from the videos, you will also need the following librairies :

- [Imageio](https://pypi.org/project/imageio/)
- [FFMPEG](https://www.ffmpeg.org/download.html)
- [face_recognition](https://github.com/ageitgey/face_recognition)

## Pretrained models

You can find the pretrained weight in the `weights` folder. The `_DF` extension correspond to a model trained to classify deepfake-generated images and the `_F2F` to Face2Face-generated images.

## References

Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018, September). [MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/abs/1809.00888). In IEEE Workshop on Information Forensics and Security, WIFS 2018.

