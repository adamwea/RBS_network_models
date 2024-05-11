# Use a base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

# Install essential packages
RUN apt-get -y update && apt-get install -y autoconf automake gcc g++ make gfortran wget curl

# Set environment variables
ENV MPICH_VERSION=4.1.1 \
    PATH=/usr/bin:/usr/local/bin:/bin:/app \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

# Install dependencies
RUN apt-get update && \
    apt-get install -y wget build-essential

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python3-openssl git 

# Download and extract Python 3.8 source code
RUN mkdir ~/python38 && \
    cd ~/python38 && \
    wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz && \
    tar -xf Python-3.8.16.tgz

# Configure the build
RUN cd ~/python38/Python-3.8.16 && \
    ./configure --enable-optimizations --enable-shared

# Compile the source code
RUN cd ~/python38/Python-3.8.16 && \
    make -j$(nproc)

# Install Python
RUN cd ~/python38/Python-3.8.16 && \
    make install

RUN apt-get update && apt-get install -y build-essential wget tar libevent-dev hwloc libhwloc-dev pandoc libfabric-dev && \
    rm -rf /var/lib/apt/lists/*

# Define the versions
#ENV PMIX_VERSION=4.1.0
ENV MPICH_VERSION=4.1.1
ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# # Install PMIx
# RUN cd /usr/local/src/ && \
#     wget https://github.com/openpmix/openpmix/releases/download/v${PMIX_VERSION}/pmix-${PMIX_VERSION}.tar.gz && \
#     tar xf pmix-${PMIX_VERSION}.tar.gz && \
#     cd pmix-${PMIX_VERSION} && \
#     ./configure --prefix=/usr/local && \
#     make -j$(nproc) && \
#     make install && \
#     cd /usr/local/src && \
#     rm -rf pmix-${PMIX_VERSION} pmix-${PMIX_VERSION}.tar.gz

RUN apt-get update && apt-get install -y build-essential wget tar libevent-dev hwloc libhwloc-dev pandoc \
    libfabric-dev libpsm-infinipath1-dev libpsm2-dev librdmacm-dev libibverbs-dev
    
# Install MPICH with PMIx support
RUN cd /usr/local/src/ && \
    wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar xf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --with-device=ch4:ofi --enable-fortran=no --enable-static=no \
    --with-pm=hydra && \
    #--with-pmi=pmix --with-pmix=/usr/local && \
    make -j6 && \
    make install && \
    cd /usr/local/src && \
    rm -rf mpich-${MPICH_VERSION} mpich-${MPICH_VERSION}.tar.gz

RUN ls -l /usr/local/bin/mpiexec
RUN mpiexec --version
# RUN ls -l /usr/local/include/pmi.h
# RUN ls -l /usr/local/lib/libpmi*

# Update OS and install packages
RUN apt-get update && \
    apt-get install --yes build-essential gfortran \ 
    #python3-dev \
    python3-pip wget && \
    apt-get clean all

# Set up for NEURON
RUN apt-get install -y locales autoconf automake gcc g++ make vim ssh git emacs aptitude build-essential xterm iputils-ping net-tools screen graphviz && \
    apt-get clean all

# Install tzdata
RUN apt-get install -y tzdata

# More libs for NEURON
RUN apt-get install -y libgraphviz-dev pkg-config expat zlib1g-dev libncurses5-dev libncursesw5-dev python3-tk && \
    apt-get clean all

# # Install necessary tools including Bison and Flex
# RUN apt-get update && apt-get install -y \
#     cmake \
#     bison \
#     flex

# Example for OpenMPI with CUDA support
# RUN apt-get install -y libopenmpi-dev
# ENV OMPI_MCA_opal_cuda_support=true

# Install user management utilities
#RUN apt-get update && apt-get install -y passwd
#RUN apt-get install adduser

# Install pmi library
#RUN apt-get install libpmi-dev

# Install user management tools
RUN apt-get update && apt-get install -y

# Create a virtual environment
RUN python3.8 -m venv /opt/venv

# Install essential Python libraries
RUN /opt/venv/bin/pip install \
    numpy \
    matplotlib h5py ruamel.yaml jupyter jupyter_server scipy six bluepyopt \
    neuron \
    netpyne Igor

# Upgrade Pillow
RUN /opt/venv/bin/pip install --upgrade Pillow

# Install additional Python packages
RUN /opt/venv/bin/pip install bokeh contextlib2 \
    cycler fonttools future jinja2 kiwisolver lfpykit \
    markupsafe matplotlib-scalebar meautility packaging \
    pandas pyparsing pytz pyyaml schema tornado inspyred

# Install mpi4py
RUN /opt/venv/bin/python -m pip install mpi4py

#Install remaining batchRun.py dependencies
# Prepare a writable directory for Fontconfig cache
RUN mkdir /opt/fontconfig && \
    chmod 777 /opt/fontconfig

# Set environment variable for Fontconfig to use the new cache directory
ENV FONTCONFIG_PATH=/opt/fontconfig

# Activate the virtual environment by default
ENV PATH="/opt/venv/bin:$PATH"

#add neuron to python path
ENV PYTHONPATH opt/venv/bin/nrniv:$PYTHONPATH

# Set environment variables for MPICH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV MPIR_CVAR_CH3_NOLOCAL=0
ENV MPIR_CVAR_CH4_NOLOCAL=0
ENV HYDRA_BOOTSTRAP=slurm
ENV RDMAV_FORK_SAFE=1

#allow write to /opt
RUN chmod -R o+w /opt \
    && fc-cache --really-force --verbose

# #give mpiuser permissions
# RUN chmod -R o+w /usr \
#     && fc-cache --really-force --verbose

#add /usr/sbin to path so that adduser is found
#ENV PATH /usr/sbin:$PATH

# Create a non-root user
#RUN groupadd -r mpiuser && useradd -r -g mpiuser -m -d /home/mpiuser -s /bin/bash mpiuser
#RUN adduser --disabled-password --gecos '' --shell /bin/bash mpiuser

# Change to non-root privilege
# RUN chown -R mpiuser:mpiuser /usr
# RUN chown -R mpiuser:mpiuser /opt
#USER mpiuser
# Set the working directory to the user's home directory
#RUN chown mpiuser:mpiuser /home/mpiuser
WORKDIR /app
# WORKDIR /opt/app


# Set default command
CMD ["bash"]

#CMD ["mpiexec", "-np", "4", "python3", "testmpi.py"]
#CMD ["mpiexec", "-np", "1", "nrniv", "-mpi", "batchRun.py"]

#without slurm
#export RDMAV_FORK_SAFE=1
#mpiexec -bootstrap fork -np 1 --verbose nrniv -mpi -python batchRun.py
#mpiexec -bootstrap fork -np 1 python3 testmpi.py
#with slurm
#mpiexec -np 1 --verbose nrniv -mpi -python batchRun.py