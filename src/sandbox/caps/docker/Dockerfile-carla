FROM nvidia/cudagl:9.0-devel-ubuntu16.04

ARG UID=1000
ARG GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gcc \
        libc6-dev \
        libglu1 \
        libglu1:i386 \
        libsm6 \
        libxv1 \
        libxv1:i386 \
        make \
        x11-xkb-utils \
        xauth \
        xfonts-base \
        xkb-data && \
    apt-get install --reinstall -y build-essential && \
    apt-get install -y \
        sudo \
        nano \
        wget \
        bzip2 \
        gcc \
        g++ \
        git \
        tmux && \
    rm -rf /var/lib/apt/lists/*


RUN groupadd -g	$GID caps-user
RUN useradd -m -u $UID -g $GID caps-user && echo "caps-user:caps" | chpasswd && adduser caps-user sudo
USER caps-user

ENV HOME /home/caps-user
WORKDIR $HOME

ENV SOURCEDIR $HOME/source
RUN mkdir $SOURCEDIR

# install miniconda
RUN cd $SOURCEDIR && \   
    wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p $SOURCEDIR/miniconda && \
    rm Miniconda3-4.5.4-Linux-x86_64.sh && \
    cd
ENV PATH $SOURCEDIR/miniconda/envs/caps/bin:$SOURCEDIR/miniconda/bin:$PATH
# setup caps miniconda env
RUN conda create -y -n caps python=3.5 && \
    echo 'source activate caps' >> ~/.bashrc
# install to caps env
RUN conda install -n caps -y cudnn==7.1.2
RUN pip install tensorflow-gpu==1.8.0
RUN pip install colorlog==3.1.0
RUN pip install pandas==0.21.0
RUN conda install -n caps -y pillow=5.0.0
RUN conda install -n caps -y matplotlib=2.2.2
RUN pip install ipython==6.4.0

# setup caps
RUN echo 'export PYTHONPATH=$PYTHONPATH:$HOME/caps/src' >> ~/.bashrc

# setup tmux
RUN echo 'set-option -g default-shell /bin/bash' >> ~/.tmux.conf

# set display
RUN echo 'export DISPLAY=:0' >> ~/.bashrc


# carla
RUN mkdir $SOURCEDIR/carla && \
    cd $SOURCEDIR/carla && \
    wget https://people.eecs.berkeley.edu/~gregoryk/downloads/CARLA_0.8.4.tar.gz && \
    tar -xvf CARLA_0.8.4.tar.gz && \
    rm CARLA_0.8.4.tar.gz && \
    cd
RUN echo 'export CARLAPATH=$HOME/source/carla' >> ~/.bashrc
RUN echo 'export PYTHONPATH=$PYTHONPATH:$HOME/source/carla/PythonClient' >> ~/.bashrc
RUN echo 'caps' | sudo -S apt-get update
RUN echo 'caps' | sudo -S apt-get -y install x11-xserver-utils libxrandr-dev
RUN echo 'caps' | sudo -S rm -rf /var/lib/apt/lists/*

