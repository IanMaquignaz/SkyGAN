# For more information, please refer to https://aka.ms/vscode-docker-python

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# ARGUMENTS
ARG PYTHON="python3.9"

# WORKING DIRECTORY
WORKDIR /workspace

# Set Timezone
ENV TZ=America/Montreal
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update the box (refresh apt-get)
# Packages can be messed up, but using '--fix-missing' results in a full docker rebuild
# This uses '--fix-missing' only if something fails.
RUN apt-get update -y --fix-missing
RUN apt-get update -y && if [ $? -ne 0 ] ; then apt-get update -y --fix-missing ; fi ;

# Install apt-get requirements
RUN apt-get install -y tmux htop
RUN apt-get install -y wget curl iputils-ping
RUN apt-get install -y zip unzip nano
RUN apt-get install -y git build-essential cmake pkg-config
RUN apt-get install -y software-properties-common
RUN apt install -y libssl-dev libffi-dev

# Install Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install -y $PYTHON ${PYTHON}-dev ${PYTHON}-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN $PYTHON get-pip.py
RUN $PYTHON -m pip install --upgrade pip
RUN pip3 install --upgrade pip
RUN update-alternatives --install /usr/bin/python python $(which $PYTHON) 1

# Install Python requirements
RUN $PYTHON -m pip install debugpy numpy tqdm

# Get python requirements
# Docker python requirements are required
COPY requirements_python.txt ./
RUN $PYTHON -m pip install -r ./requirements_python.txt

# SlackTools and Utils python requirements are optional
# But docker copy needs at least one file to exist
# SlackTools
COPY requirements_python.txt requirements_python_customLibs.txt ./
RUN test -f requirements_python_customLibs.txt && $PYTHON -m pip install -r ./requirements_python_customLibs.txt

# Update CMake
RUN $PYTHON -m pip install cmake --upgrade

# Install OpenEXR
RUN cd ~
RUN git clone --recursive https://github.com/AcademySoftwareFoundation/openexr.git
RUN cd openexr && git checkout release
RUN mkdir -p openexr/build
RUN apt-get install libghc-half-dev libopenexr-dev -y
RUN cd openexr/build && cmake .. && make -j8 && make install
RUN $PYTHON -m pip install openexr

# OpenCV
RUN apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran libatlas-base-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-dev
RUN $PYTHON -m pip install opencv-python

# Install Pillow
RUN apt-get install -y libjpeg8-dev zlib1g-dev
RUN $PYTHON -m pip install --upgrade pip Pillow

# Other requirements
COPY requirements_aptGet.txt .
RUN xargs apt-get install -y < ./requirements_aptGet.txt

# Cleanup
RUN apt-get clean autoclean
RUN apt-get autoremove --yes
# Saves ~200MB but DESTROYS APT-GET!
# RUN rm -rf /var/lib/{apt,dpkg,cache,log}/

# Create links for local installs
RUN mkdir -p ~/.local/bin
RUN echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc

# Creates a non-root user with an explicit UID
ARG USER_NAME="toor"
ARG USER_ID=5678
ARG GROUP_ID=8765
RUN groupadd -g ${GROUP_ID} docker
RUN useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}
RUN echo "${USER_NAME}:toor" |  chpasswd
RUN mkdir /local
RUN chown ${USER_ID}:${GROUP_ID} /local

USER $USER_ID:${GROUP_ID}

# Create links for local installs
RUN mkdir -p ~/.local/bin
RUN echo "" >> ~/.bashrc && echo "### CUSTOM ###" >> ~/.bashrc
RUN echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc
RUN echo "alias python=$PYTHON" >> ~/.bashrc

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["$PYTHON", "run.py"]
# CMD $PYTHON -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py
