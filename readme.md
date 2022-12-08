
The code was developed on top of the excellent [nnUNet library](https://github.com/MIC-DKFZ/nnUNet). 
Please refer to the original repo for the installation, usages, and common Q&A

## Inference with docker image
You can run the inference with the docker image that we submitted to the competition by following these instructions:

1. Install `docker-ce` and `nvidia-container-toolkit` ([instruction](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
2. Pull the docker image from [here](https://hub.docker.com/r/rixez/brats21nnunet)
3. Gather the data you want to infer on in one folder. The naming of the file should follow the convention: `BraTS2021_ID_<contrast>.nii.gz` with `contrast` being `flair, t1, t1ce, t2`
4. Run the command: ```docker run -it --rm --gpus device=0 --name nnunet -v "/your/input/folder/":"/input" -v "/your/output/folder/":"/output" rixez/brats21nnunet ```, replacing `/your/input/folder` and `/your/output/folder` with the absolute paths to your input and output folder.
5. You can find the prediction results in the specified output folder.

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

