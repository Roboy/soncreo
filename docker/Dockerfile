FROM missxa/melodic-crystal-roboy
LABEL vendor="Roboy"
LABEL vendor.maintainer="Roboy Cognition Team"
LABEL name="Roboy Soncreo"
LABEL version="1.0"
LABEL description="Docker for Speech Synthesis Library"

# Install Cuda Toolkit (NVCC) and Requirements for Soncreo
RUN apt update
RUN apt install -y build-essential dkms
RUN apt install -y gcc g++
RUN apt install -y freeglut3 freeglut3-dev libxi-dev libxmu-dev
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update
RUN apt install -y cuda-toolkit-10-0 cuda-libraries-10-0 cuda-libraries-dev-10-0
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
RUN apt update && apt install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

## nvidia-docker
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda=10.0"
RUN export CUDA_HOME=/usr/local/cuda
RUN export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
RUN export LIBRARY_PATH=/usr/local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN nvcc -V

## Installing Torch
RUN python3 -m pip install --upgrade setuptools wheel
RUN pip3 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torch

## Setting the root working dir and installing requirements
RUN mkdir -p /tts
RUN chmod 777 -R /tts
WORKDIR /tts
COPY requirements.txt /tts/requirements.txt
RUN add-apt-repository universe && sudo apt update
RUN apt install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
RUN apt install -y sox libsox-dev libsox-fmt-all
RUN pip3 install --upgrade pip
#RUN mkdir libs
#RUN cd libs
#RUN git clone https://github.com/NVIDIA/apex.git
#RUN cd apex
#RUN pip3 install --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .
#RUN cd ../../
RUN pip3 install -r requirements.txt
RUN apt -y install byobu
RUN apt update

## Setting the entrypoint script
COPY ./tts-apex-entrypoint.sh /
ENTRYPOINT ["/tts-apex-entrypoint.sh"]

