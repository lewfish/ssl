# train swav model

## local

```
python -m pytorch_models.train_swav --data_dir /opt/data/research/ssl/test-chips/ \
    --default_root_dir /opt/data/research/ssl/swav/test-output --gpus 0 --fp32 \
    --batch_size 4 --max_steps 3 --num_workers 0 --fast_dev_run 0
```

## remote

```
export JOB_DEF="lfishgoldPyTorchModelsGpuJobDefinition"
export JOB_QUEUE="lfishgoldRasterVisionGpuJobQueue"
python -m pytorch_models.train_swav \
    --data_dir s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-unsupervised-chips/4-5-2021a/chip \
    --default_root_dir s3://research-lf-dev/ssl/output-4-5-2021/baselines/swav/4-13-2021a/ \
    --max_epochs 4 \
    --batch_size 64 \
    --init_weights s3://research-lf-dev/ssl/checkpoints/swav_imagenet.pth.tar \
    --fast_dev_run 0 --aws_batch
```

### increase epochs

```
export JOB_DEF="lfishgoldPyTorchModelsGpuJobDefinition"
export JOB_QUEUE="lfishgoldRasterVisionGpuJobQueue"
python -m pytorch_models.train_swav \
    --data_dir s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-unsupervised-chips/4-5-2021a/chip \
    --default_root_dir s3://research-lf-dev/ssl/output-4-5-2021/baselines/swav/4-19-2021a/ \
    --max_epochs 20 \
    --batch_size 64 \
    --init_weights s3://research-lf-dev/ssl/checkpoints/swav_imagenet.pth.tar \
    --fast_dev_run 0 --aws_batch
```

# fine tune

## test

```
rm -r /opt/data/research/ssl/test-output/
rastervision run inprocess rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri /opt/data/research/ssl/test-output \
    -a test True \
    -a aois "paris,shanghai" \
    -a train_sz 1.0 \
    chip train
```

## chips for unsupervised baseline model

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-unsupervised-chips/4-5-2021a/ \
    --pipeline-run-name all-spacenet-chips \
    --splits 16 \
    chip
```

## baseline model: supervised train 100% on other 3 aois

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-supervised/4-7-2021a/ \
    -a aois "paris,shanghai,khartoum" \
    -a num_epochs 20 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-supervised/4-5-2021a/chip \
    --pipeline-run-name paris-shanghai-khartoum \
    --splits 8 train
```

## chips for vegas finetuning experiments

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/ \
    -a aois "vegas" \
    --pipeline-run-name vegas-chips \
    --splits 8 \
    chip
```

## finetune on vegas using imagenet supervised pretrained model

### 0.001

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/0.001/200 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.001 \
    -a num_epochs 200 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

### 0.01

0.01/50
*0.01/100
0.01/200

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/0.01/50 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.01 \
    -a num_epochs 50 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/0.01/200 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.01 \
    -a num_epochs 200 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

### 0.1

0.1/25
0.1/50
*0.1/100

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/0.1/50 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.1 \
    -a num_epochs 50 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/0.1/100 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.1 \
    -a num_epochs 100 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

### 1.0

1.0/5
1.0/10
*1.0/20
1.0/40

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/1.0/10 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 1.0 \
    -a num_epochs 10 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/1.0/20 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 1.0 \
    -a num_epochs 20 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/imagenet-supervised/4-7-2021a/1.0/40 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 1.0 \
    -a num_epochs 40 \
    --pipeline-run-name finetune-imagenet-pretrained \
    --splits 8 \
    train
```

## finetune on vegas using spacenet supervised pretrained model

### 0.001

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-supervised/4-8-2021a/0.001/200 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.001 \
    -a num_epochs 200 \
    -a init_weights s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-supervised/4-7-2021a/train/last-model.pth \
    --pipeline-run-name finetune-spacenet-pretrained \
    --splits 8 \
    train
```

### 0.01

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-supervised/4-8-2021a/0.01/100 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.01 \
    -a num_epochs 100 \
    -a init_weights s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-supervised/4-7-2021a/train/last-model.pth \
    --pipeline-run-name finetune-spacenet-pretrained \
    --splits 8 \
    train
```

### 0.1

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-supervised/4-8-2021a/0.1/50 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.1 \
    -a num_epochs 50 \
    -a init_weights s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-supervised/4-7-2021a/train/last-model.pth \
    --pipeline-run-name finetune-spacenet-pretrained \
    --splits 8 \
    train
```

### 1.0

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-supervised/4-8-2021a/1.0/20 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 1.0 \
    -a num_epochs 20 \
    -a init_weights s3://research-lf-dev/ssl/output-4-5-2021/baselines/spacenet-supervised/4-7-2021a/train/last-model.pth \
    --pipeline-run-name finetune-spacenet-pretrained \
    --splits 8 \
    train
```

## finetune on vegas using spacenet unsupervised pretrained model

## 0.001

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-unsupervised/4-15-2021a/0.001/200 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.001 \
    -a num_epochs 200 \
    -a init_weights s3://research-lf-dev/ssl/output-4-5-2021/baselines/swav/4-13-2021a/backbone2.pth \
    --pipeline-run-name finetune-spacenet-pretrained \
    --splits 8 \
    train
```

## 0.01 / unsupervised trained for more epochs

```
rastervision run batch rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_ssl \
    -a raw_uri s3://spacenet-dataset/ \
    -a spacenet_csv_uri /opt/data/research/ssl/spacenet.csv \
    -a root_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/spacenet-unsupervised/4-20-2021a/0.01/200 \
    -a chip_uri s3://research-lf-dev/ssl/output-4-5-2021/finetuned/chips/4-5-2021a/chip/ \
    -a aois "vegas" \
    -a train_sz 0.01 \
    -a num_epochs 200 \
    -a init_weights s3://research-lf-dev/ssl/output-4-5-2021/baselines/swav/4-19-2021a/backbone.pth \
    --pipeline-run-name finetune-spacenet-pretrained \
    --splits 8 \
    train
```
