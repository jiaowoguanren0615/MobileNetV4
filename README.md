<h1 align='center'>MobileNetV4</h1>

# [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)
## This project is implemented in PyTorch, can be used to train your image-datasets for vision tasks.  
## [official source code](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py)  
## For segmentation tasks, please refer this [github warehouse](https://github.com/jiaowoguanren0615/Segmentation_Factory/blob/main/models/backbones/mobilenetv4.py)  
## For detection tasks(___Based on DETR Detector architecture___), please refer this [github warehouse](https://github.com/jiaowoguanren0615/Detection-Factory/blob/main/configs/salience_detr_mobilenetv4_medium_800_1333.py)  
![image](https://github.com/jiaowoguanren0615/MobileNetV4/blob/main/sample_png/mobilenetV4.jpg)  



## Preparation

### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: MobileNetV4 Model
    ├── build_mobilenet_v4.py: Construct MobileNetV4 models
    ├── extra_attention_block.py: MultiScaleAttentionGate module
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── optimizer.py: Define Sophia/MARS optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  
If you want to add an extra MSAG(MultiScaleAttentionGate) module, set the __extra_attention_block__ parameter to True.  
Moreover, you can set the ___opt_auc___ parameter to True if you want to optimize your model for a better performance(maybe~).  

## Use Sophia Optimizer (in util/optimizer.py)
You can use anther optimizer sophia, just need to change the optimizer in ___train_gpu.py___, for this training sample, can achieve better results
```
# optimizer = create_optimizer(args, model_without_ddp)
optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=args.weight_decay)
```

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.finetune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights. For this project, I recommend that set it to ___True___.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)  

### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. 

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## ONNX Deployment
### step 1: ONNX export (modify the param of ___output___, ___model___ and ___checkpoint___)  
```bash
python onnx_export.py --model=mobilenetv4_small --output=./mobilenetv4_small.onnx --checkpoint=./output/mobilenetv4_small_best_checkpoint.pth
```

### step2: ONNX optimise
```bash
python onnx_optimise.py --model=mobilenetv4_small --output=./mobilenetv4_small_optim.onnx'
```

### step3: ONNX validate (modify the param of ___data_root___ and ___onnx-input___)  
```bash
python onnx_validate.py --data_root=/mnt/d/flower_data --onnx-input=./mobilenetv4_small_optim.onnx
```


## Citation
```
@article{qin2024mobilenetv4,
  title={MobileNetV4-Universal Models for the Mobile Ecosystem},
  author={Qin, Danfeng and Leichner, Chas and Delakis, Manolis and Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and Banbury, Colby and Ye, Chengxi and Akin, Berkin and others},
  journal={arXiv preprint arXiv:2404.10518},
  year={2024}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jiaowoguanren0615/MobileNetV4&type=Date)](https://star-history.com/#jiaowoguanren0615/MobileNetV4&Date)
