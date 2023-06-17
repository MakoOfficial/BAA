# BAA

Refactor

6/5:
changed the myKit.py,but not updata the requirements.txt

6/7
updata the myKit.py,requirements.txt,mymodel.py
add Animator.py,accuracy.ipynb for evaluating the output of model
add dir "animator" for saving the loss map

6/9
Training MAE loss is 7.37, when we delete the MMCA module , resualt is seem to be worth.So we spurpose to add the MMCA module and freeze the resnet feature net

6/13
Freeze the MMCA module

6/17
Due to the detached-training is not good, we try to use the original RseNet50 to training.Subsequencely，we don't sample the training-dataset.