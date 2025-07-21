# Base image: Ubuntu 22.04
FROM ubuntu:22.04

LABEL org.opencontainers.image.version="22.04"
LABEL org.opencontainers.image.ref.name="ubuntu"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    MPICH_VERSION=4.1.1 \
    SLURM_VERSION=22.05.8 \
    TZ=America/Los_Angeles \
    FONTCONFIG_PATH=/opt/fontconfig \
    PYTHONPATH=/opt/venv/bin/nrniv:${PYTHONPATH:-} \
    RDMAV_FORK_SAFE=1 \
    HYDRA_BOOTSTRAP=slurm \
    MPIR_CVAR_CH3_NOLOCAL=0 \
    MPIR_CVAR_CH4_NOLOCAL=0 \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/bin:/app:${PATH:-} \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

# Install system dependencies and MPI development libraries for mpi4py
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        python3-openssl \
        git \
        locales \
        autoconf \
        automake \
        g++ \
        make \
        vim \
        ssh \
        gcc \
        emacs \
        aptitude \
        xterm \
        iputils-ping \
        net-tools \
        screen \
        graphviz \
        tzdata \
        libevent-dev \
        hwloc \
        libhwloc-dev \
        libfabric-dev \
        libpsm-infinipath1-dev \
        libpsm2-dev \
        librdmacm-dev \
        libibverbs-dev \
        libslurm-dev \
        libgraphviz-dev \
        python3-tk \
        gfortran \
        libmpich-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and compile Python 3.8.16 from source
RUN mkdir ~/python38 && cd ~/python38 && \
    wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz && \
    tar -xf Python-3.8.16.tgz && cd Python-3.8.16 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && make install

# Install Python packages in a virtual environment
RUN python3.8 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install \
        numpy \
        matplotlib \
        h5py \
        ruamel.yaml \
        jupyter \
        jupyter_server \
        scipy \
        six \
        bluepyopt \
        neuron \
        netpyne \
        Igor && \
    /opt/venv/bin/pip install --upgrade Pillow

# Install additional Python packages
RUN /opt/venv/bin/pip install \
        bokeh \
        contextlib2 \
        cycler \
        fonttools \
        future \
        jinja2 \
        kiwisolver \
        lfpykit \
        markupsafe \
        matplotlib-scalebar \
        meautility \
        packaging \
        pandas \
        pyparsing \
        pytz \
        pyyaml \
        schema \
        tornado \
        inspyred \
        dill

# Install mpi4py (after installing MPI development libraries)
RUN /opt/venv/bin/pip install mpi4py

# Set up fontconfig
RUN mkdir /opt/fontconfig && chmod 777 /opt/fontconfig && fc-cache --really-force --verbose

# Download and install MPICH
RUN cd /usr/local/src/ && \
    wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar xf mpich-${MPICH_VERSION}.tar.gz && cd mpich-${MPICH_VERSION} && \
    ./configure --with-device=ch4:ofi --enable-fortran=no --enable-static=no --with-pm=hydra && \
    make -j6 && make install && cd /usr/local/src && \
    rm -rf mpich-${MPICH_VERSION} mpich-${MPICH_VERSION}.tar.gz

# Download and install SLURM
RUN cd /usr/local/src/ && \
    wget https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 && \
    tar xf slurm-${SLURM_VERSION}.tar.bz2 && cd slurm-${SLURM_VERSION} && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && make install && cd /usr/local/src && \
    rm -rf slurm-${SLURM_VERSION} slurm-${SLURM_VERSION}.tar.bz2

# Set the working directory
WORKDIR /app

# # Add submodules to the PATH
# ENV PATH="/app/submodules:$PATH"

# Set default command
CMD ["bash"]