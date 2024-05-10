# Use a base image
FROM ubuntu:latest
WORKDIR /opt

ENV DEBIAN_FRONTEND noninteractive

# Install essential packages
RUN apt-get -y update && apt-get install -y autoconf automake gcc g++ make gfortran wget curl

# Set environment variables
ENV MPICH_VERSION=4.2.1 \
    PATH=/usr/bin:/usr/local/bin:/bin:/app \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

# Install MPICH
RUN cd /usr/local/src/ && \
    wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar xf mpich-${MPICH_VERSION}.tar.gz && \
    rm mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --with-device=ch4:ofi --enable-fortran=no --enable-static=no && \
    make -j 4 && make install && \
    cd /usr/local/src && \
    rm -rf mpich-${MPICH_VERSION}

# Update OS and install packages
RUN apt-get update && \
    apt-get install --yes build-essential gfortran python3-dev python3-pip wget && \
    apt-get clean all

# Set up for NEURON
RUN apt-get install -y locales autoconf automake gcc g++ make vim ssh git emacs aptitude build-essential xterm iputils-ping net-tools screen graphviz && \
    apt-get clean all

# Install tzdata
RUN apt-get install -y tzdata

# More libs for NEURON
RUN apt-get install -y libgraphviz-dev pkg-config expat zlib1g-dev libncurses5-dev libncursesw5-dev python3-tk && \
    apt-get clean all

# Install user management tools
RUN apt-get update && apt-get install -y

# Install Python, pip and venv
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

#--
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

# Install pyenv
RUN apt-get update && \
    apt-get install -y git curl && \
    curl https://pyenv.run | bash

# Install dependencies for Python build
RUN apt-get update && apt-get install -y \
    libbz2-dev \
    libffi-dev \
    libreadline-dev \
    libssl-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    zlib1g-dev

# Set environment variables for pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Install Python 3.8 with pyenv
RUN pyenv install 3.8.12
RUN pyenv global 3.8.12

#--

# Install essential Python libraries
RUN /opt/venv/bin/pip install numpy matplotlib h5py ruamel.yaml jupyter jupyter_server scipy six bluepyopt neuron netpyne Igor

# Upgrade Pillow
RUN /opt/venv/bin/pip install --upgrade Pillow

# Install additional Python packages
RUN /opt/venv/bin/pip install bokeh contextlib2 cycler fonttools future jinja2 kiwisolver lfpykit markupsafe matplotlib-scalebar meautility packaging pandas pyparsing pytz pyyaml schema tornado

# Install mpi4py
RUN /opt/venv/bin/python -m pip install mpi4py

# Prepare a writable directory for Fontconfig cache
RUN mkdir /opt/fontconfig && \
    chmod 777 /opt/fontconfig

# Set environment variable for Fontconfig to use the new cache directory
ENV FONTCONFIG_PATH=/opt/fontconfig

# Activate the virtual environment by default
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user
#RUN useradd -m myuser -d /opt/myuser

# Set HOME environment variable
#ENV HOME=/opt/myuser

WORKDIR /app

# # Make the home directory writable
# RUN chmod a+w /opt/myuser
RUN chmod -R o+w /opt \
    && fc-cache --really-force --verbose

# # Switch to the non-root user
# USER myuser

# Set default command
CMD ["bash"]
