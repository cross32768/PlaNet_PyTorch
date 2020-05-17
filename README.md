# PlaNet_PyTorch
Unofficial re-implementation of "Learning Latent Dynamics for Planning from Pixels" (https://arxiv.org/abs/1811.04551 )

## Instructions
For training, install the requirements (see below) and run
```python
python3 train.py
```

To test learned model, run
```python
python3 test.py dir
```
To predict video with learned model, run
```python
python3 video_prediction.py dir
```
dir should be log_dir of train.py

## Requirements
* Python3
* Mujoco (for DeepMind Control Suite)

and see requirements.txt for required python library

## References
* [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
* [Official Implementation](https://github.com/google-research/planet)


## TODO
* Add results of experiments
* Add code to fix seed and run controlled experiments
* Generalize code for other environments
