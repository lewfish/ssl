FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y git wget=1.* && apt-get clean

RUN wget -q -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib/:$LD_LIBRARY_PATH
RUN conda install -y python=3.7
RUN python -m pip install --upgrade pip

RUN conda install -y notebook=6.1.* matplotlib=3.3.* scikit-learn=0.23.*
RUN pip install torch==1.7.* awscli==1.19.35 boto3==1.17.35 \
    opencv-python==4.5.1.48 torchvision==0.8.*

# From https://vissl.readthedocs.io/en/v0.1.5/installation.html
RUN apt-get -y install git
RUN mkdir /opt/vissl/ && \
    cd /opt/vissl/ && \
    git clone --recursive https://github.com/facebookresearch/vissl.git && \
    cd vissl && \
    pip install --progress-bar off -r requirements.txt && \
    pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/master && \
    pip install -e .[dev]

RUN pip install rastervision-pipeline==0.13 rastervision-aws-s3==0.13

ENV PYTHONPATH=/opt/src/:$PYTHONPATH
COPY vissl_experiments /opt/src/vissl_experiments/
COPY configs /opt/src/configs/
WORKDIR /opt/src

CMD ["bash"]
