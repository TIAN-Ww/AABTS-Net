
The code was developed on top of the excellent [nnUNet library](https://github.com/MIC-DKFZ/nnUNet). 
Please refer to the original repo for the installation, usages, and common Q&A

Pytorch 1.9.1
Cuda 11.4 
RTX3090. 

## Training with the model
You can train the models that we used for the competition using the command:
```
nnUNet_train 3d_fullres trainer_name <TASK_ID> <FOLD> --npz 
nnUNet_train 3d_fullres trainer_name <TASK_ID> <FOLD> --npz 
```


## Acknowledgement
This repo borrowed heavily from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) library and [axial attention](https://github.com/lucidrains/axial-attention)

