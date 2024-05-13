# Use a base image
FROM ubuntu:22.04

WORKDIR /opt

# Set noninteractive installation mode and timezone
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=America/Los_Angeles

# Install all necessary packages in one step, excluding python3-openssl and python3-tk
RUN apt-get update && apt-get install -y \
    autoconf automake gcc g++ make gfortran wget curl libevent-dev hwloc libhwloc-dev pandoc \
    libfabric-dev libpsm-infinipath1-dev libpsm2-dev librdmacm-dev libibverbs-dev libslurm-dev \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git \
    locales vim ssh emacs aptitude xterm iputils-ping net-tools screen graphviz tzdata \
    libgraphviz-dev pkg-config expat zlib1g-dev libncurses5-dev libncursesw5-dev && \
    apt-get clean all && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH="/usr/local/bin:/usr/bin:/bin:/app"
ENV LD_LIBRARY_PATH="/usr/local/lib"

# Install Python 3.8
RUN mkdir ~/python38 && \
    cd ~/python38 && \
    wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz && \
    tar -xf Python-3.8.16.tgz && \
    cd Python-3.8.16 && \
    ./configure --enable-optimizations --enable-shared && \
    make -j$(nproc) && \
    make install

# Create a symbolic link to the Python 3.8 binary, and check the Python version
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/local/bin/python3.8 /usr/bin/python
RUN python --version && which python

# Install python3-openssl and python3-tk
RUN apt-get update && apt-get install -y python3-openssl python3-tk && \
    apt-get clean all && rm -rf /var/lib/apt/lists/*

# Install all necessary packages in one step
RUN apt-get update && apt-get install -y \
    autoconf automake gcc g++ make gfortran wget curl libevent-dev hwloc libhwloc-dev pandoc \
    libfabric-dev libpsm-infinipath1-dev libpsm2-dev librdmacm-dev libibverbs-dev libslurm-dev \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
    locales vim ssh emacs aptitude xterm iputils-ping net-tools screen graphviz tzdata \
    libgraphviz-dev pkg-config expat zlib1g-dev libncurses5-dev libncursesw5-dev python3-tk && \
    apt-get clean all && rm -rf /var/lib/apt/lists/*

#RUN sleep infinity

# Install MPICH
ENV MPICH_VERSION=4.1.1
RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz -P /tmp/ && \
    tar xf /tmp/mpich-${MPICH_VERSION}.tar.gz -C /tmp/ && \
    cd /tmp/mpich-${MPICH_VERSION} && \
    ./configure --with-device=ch3:nemesis \
    --enable-fortran=no \
    --enable-static=no \
    --with-pm=hydra && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/mpich-${MPICH_VERSION} /tmp/mpich-${MPICH_VERSION}.tar.gz

# Install necessary packages for building NEURON
RUN apt-get update && apt-get install -y \
    wget curl git \
    cmake bison flex \
    build-essential \
    libncurses-dev \
    libreadline-dev \
    libx11-dev \
    python3-dev python3-pip \
    libopenmpi-dev

RUN pip install 'cython<3'

# Install C and C++ compilers that support C++17
RUN apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Clone the NEURON repository
RUN git clone https://github.com/neuronsimulator/nrn.git /opt/nrn && \
    cd /opt/nrn && \
    git checkout tags/8.0.0  # Adjust the tag as per your specific version requirement

# Configure NEURON using CMake
RUN mkdir /opt/nrn/build && cd /opt/nrn/build && \
    cmake .. \
    -DNRN_ENABLE_INTERVIEWS=OFF \
    -DNRN_ENABLE_PYTHON=ON \
    -DNRN_ENABLE_CORENEURON=ON \
    -DPYTHON_EXECUTABLE="/usr/local/bin/python3.8" \
    -DNRN_ENABLE_MPI=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/neuron \
    -DCMAKE_PREFIX_PATH="/usr/local/ncurses"

# Build and install NEURON
RUN cd /opt/nrn/build && \
    make -j$(nproc) && \
    make install

# Set environment variables
ENV PATH="/opt/neuron/bin:$PATH" \
    PYTHONPATH="/opt/neuron/lib/python"

#NEURON will try to find MPI library in the standard library paths (e.g. /usr/lib). But on some systems MPI libraries may not be in standard path. 
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python -m pip install \
    numpy matplotlib h5py ruamel.yaml jupyter jupyter_server scipy six bluepyopt \
    netpyne Pillow bokeh contextlib2 cycler fonttools future jinja2 kiwisolver lfpykit \
    markupsafe matplotlib-scalebar meautility packaging pandas pyparsing pytz pyyaml schema tornado inspyred 
    
#install mpi4py
RUN python -m pip install mpi4py

# Create a writable directory for Fontconfig cache and other usage
RUN mkdir /opt/fontconfig /app && \
    chmod -R 777 /opt/fontconfig /app

# Set environment variable for Fontconfig
ENV FONTCONFIG_PATH=/opt/fontconfig

WORKDIR /app

# Set default command
CMD ["bash"]