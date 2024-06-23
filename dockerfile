# Use the official miniconda3 base image
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists

#  copy the current directory contents into the container at /app
COPY . /workspace

# clone dassl and install requirements
RUN git clone https://github.com/KaiyangZhou/Dassl.pytorch.git

RUN pip install -r Dassl.pytorch/requirements.txt

RUN cd Dassl.pytorch && pip install -e .

# go to the working directory
WORKDIR /workspace






