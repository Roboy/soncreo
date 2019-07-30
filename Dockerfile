FROM missxa/melodic-dashing-roboy

# CUDA 10.0 is not officially supported on ubuntu 18.04
# RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
#    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
#    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/# # # nvidia-ml.list && \
#    apt-get purge --autoremove -y curl && \
#    rm -rf /var/lib/apt/lists/*

# Install Cuda Toolkit (NVCC) and Requirements for Soncreo
# RUN apt-get update
# RUN apt-get install -y build-essential dkms
# RUN apt-get install -y freeglut3 freeglut3-dev libxi-dev libxmu-dev
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN apt-get update
# RUN apt-get install -y cuda-toolkit-10-0 cuda-libraries-10-0 cuda-libraries-dev-10-0

# ENV CUDA_VERSION 10.0.130

# ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-cudart-$CUDA_PKG_VERSION && \
#     ln -s cuda-10.0 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*

# # nvidia-docker 1.0
# LABEL com.nvidia.volumes.needed="nvidia_driver"
# LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda=10.0"

# RUN export CUDA_HOME=/usr/local/cuda
# RUN export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
# RUN export LIBRARY_PATH=/usr/local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}
# RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# RUN nvcc -V

RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.0.130

ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-0 && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

ENV NCCL_VERSION 2.4.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda10.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.0 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs



RUN python3 -m pip install --upgrade setuptools wheel
RUN pip3 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torch

#After building:
RUN mkdir -p /tts
RUN chmod 777 -R /tts
WORKDIR /tts

COPY requirements.txt /tts/requirements.txt
RUN apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN add-apt-repository universe && apt update
RUN apt install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
RUN apt install -y sox libsox-dev libsox-fmt-all
RUN apt install -y gcc g++
RUN mkdir libs
RUN cd libs
RUN python3 --version
RUN apt install -y build-essential
RUN pip3 install --upgrade pip

RUN rm -rf apex

RUN cd ../
RUN pip3 install -r requirements.txt
RUN apt-get -y install byobu
RUN apt-get update

RUN pwd


RUN pip2 install pydub google-cloud-texttospeech 
RUN pip install --upgrade numpy tensorboardX
RUN apt update && apt install python-pyaudio

COPY ./keys.json /
ENV GOOGLE_APPLICATION_CREDENTIALS=/keys.json

RUN cd ~/melodic_ws/src/roboy_communication && git pull && cd ../.. && . /opt/ros/melodic/setup.sh && catkin_make

RUN git clone https://github.com/NVIDIA/apex.git && cd apex
RUN cd apex && pip3 install --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .

RUN apt install python3-pip python3-yaml
RUN pip3 install rospkg catkin_pkg

# COPY ./tts-apex-entrypoint.sh /
# ENTRYPOINT ["/tts-apex-entrypoint.sh"]
#CMD ["ros"]
