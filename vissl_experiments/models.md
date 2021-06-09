# 5/26/2021

* resisc 1 gpu test
```
export JOB_DEF="lfishgoldPyTorchModelsGpuJobDefinition"
export JOB_QUEUE="lfishgoldRasterVisionGpuJobQueue"
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_moco \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021a \
    --dataset-uri s3://research-lf-dev/ssl/datasets/resisc45.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/checkpoints/moco.torch
```

* resisc linear eval 1 gpu test with above moco model
* 91.9% test acc
```
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_moco_linear \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021b \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021a/model_final_checkpoint_phase19.torch
```

* resisc linear eval 1 gpu test with imagenet moco model
* 89.0% test acc
```
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_moco_linear \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021c \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/checkpoints/moco.torch
```

* resisc 4 gpu moco
* 35 minutes
* train loss 7.3
```
export JOB_DEF="lfishgoldPyTorchModelsQuadGpuJobDefinition"
export JOB_QUEUE="lfishgoldQuadGpuJobQueue"
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_moco \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021d \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/checkpoints/moco.torch
export JOB_DEF="lfishgoldPyTorchModelsGpuJobDefinition"
export JOB_QUEUE="lfishgoldRasterVisionGpuJobQueue"
```

* resisc linear eval 1 gpu test with above quad/50 moco model / 50 epochs
* 92.0% test acc
* 1:35 (ie. ~1.5 hours)
```
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_moco_linear \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021e \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021d/model_final_checkpoint_phase49.torch
```

* resisc linear eval 1 gpu test with imagenet moco model / 50 epochs
* 90.3% test acc
```
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_moco_linear \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021f \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/checkpoints/moco.torch
```

* resisc linear eval 1 gpu test with supervised imagenet model / 50 epochs
* 88.8% test acc
```
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config resisc_supervised_linear \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021g \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri https://download.pytorch.org/models/resnet50-19c8e357.pth
```

* resisc linear eval 1 gpu test with above single/20 moco model / 50 epochs
* 92.7% test acc
```
python -m pytorch_models    .vissl_wrapper \
    --aws-batch \
    --config resisc_moco_linear \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021h \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/resisc45-split.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/test-output/05-26-2021a/model_final_checkpoint_phase19.torch
```

# 5/27/2021

* histo moco
* 7 epochs oops
```
export JOB_DEF="lfishgoldPyTorchModelsGpuJobDefinition"
export JOB_QUEUE="lfishgoldRasterVisionGpuJobQueue"
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config histo_moco_single \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-27-2021a \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/histo-chips.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/checkpoints/moco.torch
```

* 25 epochs
```
export JOB_DEF="lfishgoldPyTorchModelsGpuJobDefinition"
export JOB_QUEUE="lfishgoldRasterVisionGpuJobQueue"
python -m pytorch_models.vissl_wrapper \
    --aws-batch \
    --config histo_moco_single \
    --output-uri s3://research-lf-dev/ssl/vissl/test-output/05-27-2021b \
    --dataset-uri s3://research-lf-dev/ssl/vissl/datasets/histo-chips.zip \
    --pretrained-uri s3://research-lf-dev/ssl/vissl/checkpoints/moco.torch
```
