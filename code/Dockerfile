# This image attempts to give access to the ControlNet model. To do so,
# the strategy is as follows:
# The main requirements are set by the model and in particular the
# Repository: https://github.com/lllyasviel/ControlNet.git
#
# 1. Clone repository
# 2. Create main portion of the environment (conda) by using the environment file
# 3. Copy over files from the folder:
#    - Copy model file (this image does not download it, many build might lock the user out#
#    - Copy over FAST api code and model load and processing portion
# 4. Black magic: configure the repo to use CPU and not GPU

# Base sytem and updates
FROM continuumio/miniconda3
RUN apt-get update
RUN apt-get upgrade -y

WORKDIR /app

## Make RUN commands use `bash --login`:
#SHELL ["/bin/bash", "--login", "-c"]

ENV PYTHONPATH=./maincode/ControlNet

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# 1. Clone repository
RUN git clone https://github.com/lllyasviel/ControlNet.git $PYTHONPATH/

# 2. Create the environment: Use cloned repo
# RUN conda create -n control python=3.8.5
RUN conda env create -f  maincode/ControlNet/environment.yaml
# activate conda
RUN echo "conda activate control" >> ~/.bashrc
ENV PATH="/opt/conda/envs/control/bin:$PATH"
ENV CONDA_DEFAULT_ENV=control

# My machine does not have Cuda and for some weird reason
# ROCm does not work wither. And without the rocm version I an not load the model
# I get a "core dumped error"
RUN pip install torch==1.12.1+rocm5.1.1 torchvision==0.13.1+rocm5.1.1 torchaudio==0.12.1 --extra-index-url  https://download.pytorch.org/whl/rocm5.1.1
# force the automatic download of other requriements from ControlNet

# 3. Copy over files and new code
#     Model: This needs to be provided when building the image
COPY control_sd15_canny.pth $PYTHONPATH/models/control_sd15_canny.pth
#     There are some dependencies required to be downloaded the 1st time it
#     Running in here, downloads it for the image, so they are available later on
#     6 fiels with one going up to 1.5GB
COPY new_code/ $PYTHONPATH
COPY engineering/img/mri_brain.jpg ./maincode/ControlNet/test_imgs/
RUN python $PYTHONPATH/awesomer_demo.py
#      myfastapy: fastAPI code
#      model load and run: awesomer_demo.py


# install the FastApi requirements
RUN pip install $PYTHONPATH/myfastapi

# 4. Black magic: In order to make the code workable with just CPU
# I apply a diff that modifies the original repo
# In order to make the project run without GPU (as above, Id o not have nvidia cards)
# I modified the original project to run just with CPU.
# Some changes are not required at all... I know
COPY diffs/gpu2cpu.diff $PYTHONPATH
RUN  cd $PYTHONPATH && git apply gpu2cpu.diff

EXPOSE 8080
CMD ["fastapi", "run", "./maincode/ControlNet/myfastapi/myawesomedemo/main.py", "--host", "0.0.0.0", "--port", "8080"]
