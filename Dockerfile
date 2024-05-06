FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        wget              && \
    apt-get clean all

WORKDIR /opt


ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    #./configure                                                             && \
    ./configure FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

# We need conda here since pip numpy is built with open blas. We want mkl.
# Install miniconda
ENV installer=Miniconda3-py38_4.9.2-Linux-x86_64.sh

RUN wget https://repo.anaconda.com/miniconda/$installer && \
    /bin/bash $installer -b -p /opt/miniconda3          && \
    rm -rf $installer

ENV PATH=/opt/miniconda3/bin:$PATH

# Install Python 3.9
#RUN /opt/miniconda3/bin/conda install python=3.9 -y

RUN /opt/miniconda3/bin/conda install numpy mpi4py -y
RUN /sbin/ldconfig

#
RUN /opt/miniconda3/bin/conda install pip -y
RUN /opt/miniconda3/bin/pip install neuron
RUN /opt/miniconda3/bin/pip install netpyne
RUN /opt/miniconda3/bin/pip install inspyred

