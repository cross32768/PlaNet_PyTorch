# PlaNet_PyTorch
Unofficial re-implementation of "Learning Latent Dynamics for Planning from Pixels" (https://arxiv.org/abs/1811.04551 )

## Instructions
For training, install the requirements (see below) and run (default environment is cheetah run)
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
dir should be log_dir of train.py and you need to specify environment corresponding to the log by arguments.



## Requirements
* Python3
* Mujoco (for DeepMind Control Suite)

and see requirements.txt for required python library

## Qualitative tesult
Example of predicted video frame by learned model
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/video_prediction.gif)

## Quantitative result
### cheetah run
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/cheetah_run.png)

### walker walk
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/walker_walk.png)

Work in progress.

I'm going to add result of experiments at least three times for each environment in the original paper.

Now my GPU is working hard!

## References
* [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
* [Official Implementation](https://github.com/google-research/planet)


## TODO
* speed up training
* Add quantitative results and more qualitative results
* Generalize code for other environments
