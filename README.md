# Falcon
Falcon is an efficient implementation of large scale distributed DNN trainer for spase id features. Take Deep & Wide models for advertisement or recommendation click through rate prediction for example, computational advertisement will use billions of sparse id features to modeling user response, like user id, advertiser id, os versions, regions, etcs.

We will always mapping each feature into different ids, however, it's tedious and computation cost to assign a unique id for each feature, especially when the training data is huge which is common for CTR tasks. In this project, we implemented a custom operation to extend tensorflow so that it can auto hash a raw feature into a unique id when training on the fly, so we don't need to assign the ids before training.

## DNN with large scale sparse id features
Like Deep & Wide or DeepFM networks, sparse id features are always seperated into different groups

![feature group](./images/feature_group.png)

## how to build this project for different enviroments

only tensorflow 1.x is supported, it will have problem with 2.x eager execution. We implemented custom op (a.k.a feature_hash_op) to extend tensorflow, the build pipeline is based on tensorflow CustomOp project, see it for more details.

After build successfully, you can wrap the custom op .so file into a pip package so that you can distributed it, or you can just copy the .so file for using. However, the .so file must be compatible with the tensorflow version you choosed when building.

1. set up the docker for building enviroment

2. git clone this repo

3. select target tensorflow version to build this project for it

4. (optional) build pip package

## example of how to use this project for training DNN models

### generate training data

### what's the format of raw features

### how to use the custom op to hash features into ids

### how to pooling on features of the same group

### distributed training

#### install horovod