FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        wget              && \
        #libreadline-dev      \
    apt-get clean all

WORKDIR /opt

# Install Python 3.9
# Update and install Python 3.9 and other essential packages
ARG python_version=3.9
# Install necessary tools and libraries
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y build-essential gfortran wget libtool libx11-dev python$python_version python$python_version-distutils
# Set Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$python_version 1 && \
    update-alternatives --set python3 /usr/bin/python$python_version

# Check python version
RUN python3 --version

#ARG mpich=3.3
ARG mpich=4.1.2
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    #./configure FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch && \
    make -j 6                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

# Install NEURON dependencies
RUN apt-get update && \
    apt-get clean && \
    apt-get install -y --fix-missing bison cmake flex git \
    libncurses-dev libopenmpi-dev libx11-dev \
    libxcomposite-dev openmpi-bin python3-dev \
    libreadline-dev
    #libxext-dev libncurses-dev \

# ldconfig is needed to update the shared library cache    
RUN /sbin/ldconfig

# We need conda here since pip numpy is built with open blas. We want mkl.
# Install miniconda
ENV installer=Miniconda3-py39_4.9.2-Linux-x86_64.sh
RUN wget https://repo.anaconda.com/miniconda/$installer && \
    /bin/bash $installer -b -p /opt/miniconda3          && \
    rm -rf $installer
ENV PATH=/opt/miniconda3/bin:$PATH

# Install Python packages from nrn_requirements.txt
#RUN conda install -c conda-forge scipy numpy mpi4py cython matplotlib -y
RUN conda install -c conda-forge scipy numpy cython matplotlib -y

#install mpi4py
RUN /opt/miniconda3/bin/conda install -c conda-forge mpi4py openmpi -y

#nrn dependencies
RUN /opt/miniconda3/bin/conda install pip -y
RUN /opt/miniconda3/bin/pip install wheel
RUN /opt/miniconda3/bin/pip install setuptools
RUN /opt/miniconda3/bin/pip install setuptools_scm
RUN /opt/miniconda3/bin/pip install scikit-build
RUN /opt/miniconda3/bin/pip install matplotlib
# bokeh 3 seems to break docs notebooks
RUN /opt/miniconda3/bin/pip install 'bokeh<3'
RUN /opt/miniconda3/bin/pip install ipython
RUN /opt/miniconda3/bin/pip install 'cython<3'
RUN /opt/miniconda3/bin/pip install packaging
# potential bug from 8.2.0 due to parallelism?
RUN /opt/miniconda3/bin/pip install 'pytest<=8.1.1' 
RUN /opt/miniconda3/bin/pip install pytest-cov
# RUN /opt/miniconda3/bin/pip install mpi4py
# RUN /opt/miniconda3/bin/pip install numpy
RUN /opt/miniconda3/bin/pip install find_libpython

# # Clone the NEURON repository
# RUN apt-get install -y git
# RUN git clone https://github.com/neuronsimulator/nrn.git /opt/nrn
# WORKDIR /opt/nrn
# # Create a build directory
# RUN mkdir build && cd build 
# #&& make clean
# # Configure NEURON with CMake
# RUN apt-get install -y cmake
# RUN cd build && \
#     cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=$HOME/neuron
# # Build and install NEURON
# RUN cd build && \
#     make -j 4 && \
#     make install
# ENV PATH=$HOME/neuron/bin:$PATH
    
RUN /opt/miniconda3/bin/pip install neuron
RUN /opt/miniconda3/bin/pip install netpyne
RUN /opt/miniconda3/bin/pip install inspyred