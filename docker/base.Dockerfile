
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG workspace_path

# COPY ${workspace_path}/docker_requirement.txt /docker_requirement.txt
ENV DEBIAN_FRONTEND=noninteractive

# basic installation for docker development 
RUN apt-get update
RUN apt-get install -y x11-apps
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
# boost install
RUN apt-get install -y libboost-all-dev		



# installation for pytorch development
RUN pip install -U scikit-learn
RUN pip install pandas
# RUN pip install madgrad             # option
RUN pip install matplotlib
RUN pip install torchsummary
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install configargparse          
RUN pip install imageio-ffmpeg
# RUN pip install loguru                  # option
RUN pip install stories
RUN pip install torchsummaryX
RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install torch-tb-profiler
# RUN pip install einops                  # option
RUN pip install mmcv-full               # option
RUN pip install timm
# RUN pip install torchinfo               # option
RUN pip install flopco-pytorch 
# RUN pip install attr                    # option
RUN pip install apex
# RUN pip install gdown                   # option
# RUN pip install unzip                   # option
RUN pip install wandb
