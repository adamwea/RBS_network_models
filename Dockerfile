# Use a base image
FROM phusion/baseimage:jammy-1.0.0

WORKDIR /opt

# Set noninteractive installation mode and timezone
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=America/Los_Angeles

# Install necessary packages including UCX
RUN apt-get update && \
    apt-get install -y \
        locales \
        wget \
        gcc \
        g++ \
        build-essential \
        libncurses-dev \
        python3 \
        python3-pip \
        libpython3-dev \
        openmpi-bin \
        openmpi-common \
        libopenmpi-dev \
        git \
        cmake \
        bison \
        libreadline-dev \
        flex \
        libucx-dev \
        libucx0 \
        libfabric-dev \
        infiniband-diags libibverbs-dev librdmacm-dev \
        libx11-dev libxcomposite-dev && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    pip3 install --upgrade pip && \
    pip3 install cython

# Build Open MPI 5.x with OpenSHMEM and UCX support
RUN mkdir -p /opt/src && cd /opt/src && \
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.1.tar.gz && \
    tar -xzf openmpi-5.0.1.tar.gz && cd openmpi-5.0.1 && \
    ./configure \
    --prefix=/opt/openmpi \
    --with-pmix \
    --with-ucx \
    && \
    make -j $(nproc) && make install

# Set environment variables for Open MPI
ENV PATH="/opt/openmpi/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/openmpi/lib:$LD_LIBRARY_PATH"

# Set PKG_CONFIG_PATH for pkg-config
ENV PKG_CONFIG_PATH="/opt/openmpi/lib/pkgconfig:$PKG_CONFIG_PATH"

# Ensure Open MPI libraries are always linked
ENV LDFLAGS="-L/opt/openmpi/lib -loshmem"
ENV CPPFLAGS="-I/opt/openmpi/include"

#Allow OpenMPI to fork processes
ENV RDMAV_FORK_SAFE=1

# Verify oshrun installation
RUN which oshrun

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH /opt/miniconda/bin:$PATH

# Create a Python 3.8 environment
RUN conda create -n netpyne_env python=3.8

# Create a user
RUN useradd -m mpiuser
RUN /sbin/ldconfig
USER mpiuser
ENV PATH=$PATH:/home/mpiuser/.local/bin

# Activate the environment
SHELL ["conda", "run", "-n", "netpyne_env", "/bin/bash", "-c"]
RUN source activate netpyne_env

RUN python3 -m pip install numpy
RUN python3 -m pip show numpy | grep Location

RUN python3 -m pip install neuron
RUN python3 -m pip show neuron | grep Location
RUN python3 -m pip install \
    numpy matplotlib h5py ruamel.yaml jupyter jupyter_server scipy six bluepyopt \
    netpyne Pillow 'bokeh<3' contextlib2 cycler fonttools future jinja2 kiwisolver lfpykit \
    markupsafe matplotlib-scalebar meautility packaging pandas pyparsing pytz pyyaml schema tornado inspyred \
    wheel setuptools setuptools_scm scikit-build ipython packaging 'pytest<=8.1.1' pytest-cov find_libpython

# Install mpi4py
RUN python3 -m pip install mpi4py

WORKDIR /app

ENV LANG en_US.utf8

CMD ["bash"]
