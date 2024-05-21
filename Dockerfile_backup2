# Use a base image
#FROM ubuntu:20.04
FROM phusion/baseimage:focal-1.2.0

WORKDIR /opt

# Set noninteractive installation mode and timezone
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=America/Los_Angeles

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
        locales \
        wget \
        build-essential \
        libncurses-dev \
        python3 \
        python3-pip \
        libpython3-dev \
        git \
        cmake \
        bison \
        libreadline-dev \
        flex \
        ssh \
        gfortran \
        libblas-dev liblapack-dev \
        libfabric-dev \
        infiniband-diags libibverbs-dev librdmacm-dev \
        libx11-dev libxcomposite-dev \
        libevent-dev \
        hwloc \
        libhwloc-dev \
        pkg-config \
        bzip2 && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    pip3 install --upgrade pip && \
    pip3 install cython && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone, build, and install PMIx
RUN git clone https://github.com/openpmix/openpmix.git /tmp/openpmix && \
    cd /tmp/openpmix && \
    git submodule update --init --recursive && \
    ./autogen.pl && \
    ./configure --prefix=/usr/local/pmix && \
    #make -j$(nproc) && \
    make -j4 && \
    make install && \
    rm -rf /tmp/openpmix

# Clone, build, and install PRRTE
RUN git clone https://github.com/openpmix/prrte.git /tmp/prrte && \
    cd /tmp/prrte && \
    git submodule update --init --recursive && \
    ./autogen.pl && \
    ./configure --prefix=/usr/local/prrte --with-pmix=/usr/local/pmix && \
    #make -j$(nproc) && \
    make -j4 && \
    make install && \
    rm -rf /tmp/prrte

# # Clone, build, and install Open MPI
# RUN git clone https://github.com/open-mpi/ompi.git /tmp/ompi && \
#     cd /tmp/ompi && \
#     git submodule update --init --recursive && \
#     ./autogen.pl && \
#     ./configure --prefix=/usr/local/ompi --with-pmix=/usr/local/pmix --with-prrte=/usr/local/prrte && \
#     make -j$(nproc) && \
#     make install && \
#     rm -rf /tmp/ompi

RUN apt-get update && \
    apt-get install -y slurm-wlm libslurm-dev slurmctld slurmd munge && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the configuration files to the appropriate locations
COPY slurm.conf /etc/slurm-llnl/slurm.conf
COPY munge.key /etc/munge/munge.key

# Set up munge key
RUN chown munge:munge /etc/munge/munge.key && \
    chmod 400 /etc/munge/munge.key

# # Start munge and slurm services
RUN /etc/init.d/munge start && \
    /etc/init.d/slurmctld start && \
    /etc/init.d/slurmd start
#     tail -f /dev/null

# Install UCX dependencies
RUN apt-get update && \
    apt-get install -y \
        binutils-dev \
        libnuma-dev \
        wget && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for PMIx
ENV PATH="/usr/local/pmix/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/pmix/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/usr/local/pmix/lib/pkgconfig:$PKG_CONFIG_PATH"

# Download, build, and install UCX
RUN wget https://github.com/openucx/ucx/releases/download/v1.12.1/ucx-1.12.1.tar.gz && \
    tar -xzf ucx-1.12.1.tar.gz && \
    cd ucx-1.12.1 && \
    ./configure --prefix=/usr/local/ucx && \
    make -j4 && \
    make install && \
    cd .. && \
    rm -rf ucx-1.12.1 ucx-1.12.1.tar.gz

# Set environment variables for UCX
ENV PATH="/usr/local/ucx/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/usr/local/ucx/lib/pkgconfig:$PKG_CONFIG_PATH"

#confirm ucx version is 1.12.1
#RUN ucx_info -v

#make docker build hang for debug purposes
#RUN tail -f /dev/null

# Clone, build, and install OpenSHMEM with UCX support
RUN git clone https://github.com/openshmem-org/osss-ucx.git /tmp/osss-ucx && \
    cd /tmp/osss-ucx && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/openshmem-ucx \
                --with-ucx=/usr/local/ucx \
                --enable-option-checking && \
    make -j4 && \
    make install && \
    rm -rf /tmp/osss-ucx

# Set the necessary environment variables
ENV PATH="/usr/local/openshmem-ucx/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/openshmem-ucx/lib:$LD_LIBRARY_PATH"
ENV C_INCLUDE_PATH="/usr/local/openshmem-ucx/include:$C_INCLUDE_PATH"

# Clone, build, and install Open MPI
RUN git clone https://github.com/open-mpi/ompi.git /tmp/ompi && \
    cd /tmp/ompi && \
    git submodule update --init --recursive && \
    ./autogen.pl && \
    ./configure --prefix=/usr/local/ompi \
                --with-pmix=/usr/local/pmix \
                --with-prrte=/usr/local/prrte \
                --with-slurm \
                #--with-ucx=/usr/local/ucx \
                #--with-shmem=/usr/local/openshmem-ucx \
                #--with-pmi \
                &&\
    #make -j$(nproc) && \
    make -j4 && \
    make install && \
    rm -rf /tmp/ompi

# # Download, build, and install PMIx
# RUN wget https://github.com/openpmix/openpmix/releases/download/v4.2.8/pmix-4.2.8.tar.bz2 && \
#     tar -xjf pmix-4.2.8.tar.bz2 && \
#     cd pmix-4.2.8 && \
#     ./configure --prefix=/usr/local/pmix && \
#     make -j$(nproc) && \
#     make install && \
#     cd .. && rm -rf pmix-4.2.8 pmix-4.2.8.tar.bz2

# # Download, build, and install PRRTE
# RUN wget https://github.com/openpmix/prrte/releases/download/v3.0.4/prrte-3.0.4.tar.bz2 && \
#     tar -xjf prrte-3.0.4.tar.bz2 && \
#     cd prrte-3.0.4 && \
#     ./configure --prefix=/usr/local/prrte --with-pmix=/usr/local/pmix && \
#     make -j$(nproc) && \
#     make install && \
#     cd .. && rm -rf prrte-3.0.4 prrte-3.0.4.tar.bz2

# # Download, build, and install Open MPI
# RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.2.tar.gz && \
#     tar -xzf openmpi-5.0.2.tar.gz && \
#     cd openmpi-5.0.2 && \
#     ./configure --prefix=/usr/local/ompi --with-pmix=/usr/local/pmix --with-prrte=/usr/local/prrte && \
#     make -j$(nproc) && \
#     make install && \
#     cd .. && rm -rf openmpi-5.0.2 openmpi-5.0.2.tar.gz

# Set environment variables for Open MPI, PMIx, and PRRTE
ENV PATH="/usr/local/pmix/bin:/usr/local/prrte/bin:/usr/local/ompi/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/pmix/lib:/usr/local/prrte/lib:/usr/local/ompi/lib:$LD_LIBRARY_PATH"
ENV C_INCLUDE_PATH="/usr/local/ompi/include:$C_INCLUDE_PATH"


#Set MpiDefault to pmix
#RUN echo "MpiDefault=pmix" >> /etc/slurm/slurm.conf

# # Clone the NEURON repository
# RUN git clone https://github.com/neuronsimulator/nrn.git /opt/nrn && \
#     cd /opt/nrn && \
#     git checkout tags/8.2.4  # Adjust the tag as per your specific version requirement

# # Configure NEURON using CMake
# RUN mkdir /opt/nrn/build && cd /opt/nrn/build && \
#     cmake .. \
#     -DNRN_ENABLE_INTERVIEWS=OFF \
#     -DNRN_ENABLE_PYTHON=ON \
#     -DNRN_ENABLE_CORENEURON=ON \
#     -DPYTHON_EXECUTABLE="/usr/bin/python3" \
#     -DNRN_ENABLE_MPI=ON \
#     -DCMAKE_INSTALL_PREFIX="/opt/neuron"
#     #-DCMAKE_PREFIX_PATH="/usr/local/ncurses"

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH /opt/miniconda/bin:$PATH

# Create a Python 3.9 environment
RUN conda create -n netpyne_env python=3.8

#create a user
RUN useradd -m mpiuser
RUN /sbin/ldconfig
USER mpiuser
ENV PATH=$PATH:/home/mpiuser/.local/bin

# Activate the environment
SHELL ["conda", "run", "-n", "netpyne_env", "/bin/bash", "-c"]
RUN source activate netpyne_env

#RUN python3 -m pip install numpy
# RUN cd /opt/nrn \
#     && python3 setup.py install

RUN python3 -m pip install numpy
RUN python3 -m pip show numpy | grep Location

RUN python3 -m pip install neuron
RUN python3 -m pip show neuron | grep Location
RUN python3 -m pip install \
    numpy matplotlib h5py ruamel.yaml jupyter jupyter_server scipy six bluepyopt \
    netpyne Pillow 'bokeh<3' contextlib2 cycler fonttools future jinja2 kiwisolver lfpykit \
    markupsafe matplotlib-scalebar meautility packaging pandas pyparsing pytz pyyaml schema tornado inspyred \
    wheel setuptools setuptools_scm scikit-build ipython packaging 'pytest<=8.1.1' pytest-cov find_libpython
    
#install mpi4py

RUN python3 -m pip install mpi4py

WORKDIR /app

ENV LANG en_US.utf8

# Add the path to shmem.h to C_INCLUDE_PATH
#ENV C_INCLUDE_PATH="/usr/include/hwloc:$C_INCLUDE_PATH"

# ENV PATH=/usr/bin:$PATH
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/openmpi/lib
# ENV OPAL_PREFIX=/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal

# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/mpi/lib
# ENV OPAL_PREFIX=/opt/hpcx/ompi
#ENV PATH $PATH:/opt/nrn/x86_64/bin

#RUN cat /etc/slurm/slurm.conf
# Set environment variables for Open MPI
#ENV OMPI_MCA_orte_rsh_agent=ssh
#ENV OMPI_MCA_plm_rsh_agent=ssh
#ENV OMPI_MCA_orte_base_help_aggregate=0

#start environment
#RUN /bin/bash -c ["source", "activate", "netpyne_env"]

#RUN source activate netpyne_env

# # Copy the startup script into the image
# COPY start-services.sh /usr/local/bin/

# # Make the script executable
# RUN chmod +x /usr/local/bin/start-services.sh

# # Use the script as the entrypoint
# ENTRYPOINT ["start-services.sh"]

# Set default command
CMD ["bash"]
# Set default command
# Create the script and make it executable
# RUN echo -e '#!/bin/bash\nsource /opt/miniconda/bin/activate netpyne_env\nexec /bin/bash' > /usr/local/bin/entrypoint.sh && \
#     chmod +x /usr/local/bin/entrypoint.sh

# # Set the script as the default command
# CMD ["/usr/local/bin/entrypoint.sh"]