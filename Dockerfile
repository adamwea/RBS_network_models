FROM python:3.8

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        wget \
        libtool \
        libx11-dev \
        bison \
        cmake \
        flex \
        git \
        libncurses-dev \
        libopenmpi-dev \
        libx11-dev \
        libxcomposite-dev \
        openmpi-bin \
        libreadline-dev \
        fontconfig \
        libdbus-1-dev \
        libdbus-glib-1-dev \
        dbus \
        #gobject-introspection \
        # meson \
        # ninja-build \
        libgirepository1.0-dev \
        python3-gi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN wget http://ftp.gnome.org/pub/GNOME/sources/gobject-introspection/1.70/gobject-introspection-1.70.0.tar.xz && \
#     tar -xf gobject-introspection-1.70.0.tar.xz && \
#     cd gobject-introspection-1.70.0 && \
#     meson setup _build && \
#     ninja -C _build && \
#     ninja -C _build install

RUN apt-get update

WORKDIR /opt

# Install Python packages using pip
RUN pip install --no-cache-dir \
    anyio \
    argon2-cffi \
    argon2-cffi-bindings \
    arrow \
    asttokens \
    attrs \
    backcall \
    beautifulsoup4 \
    bleach \
    bluepyopt \
    bokeh \
    cffi \
    comm \
    contextlib2 \
    contourpy \
    cycler

RUN pip install --no-cache-dir \    
    dbus-python \
    deap \
    debugpy \
    decorator \
    defusedxml \
    distro \
    efel \
    entrypoints \
    executing \
    fastjsonschema \
    fonttools \
    fqdn \
    future \
    h5py \
    idna \
    igor \
    ipykernel \
    ipyparallel \
    ipython \
    ipython-genutils \
    ipywidgets \
    isoduration \
    jedi \
    Jinja2 \
    jsonpointer \
    jsonschema \
    jupyter \
    jupyter-console \
    jupyter-events \
    jupyter_client \
    jupyter_core \
    jupyter_server \
    jupyter_server_terminals \
    jupyterlab-pygments \
    jupyterlab-widgets \
    kiwisolver \
    LFPykit \
    MarkupSafe \
    matplotlib \
    matplotlib-inline \
    matplotlib-scalebar \
    MEAutility \
    mistune

RUN pip install --no-cache-dir \
    mpi4py \
    nbclassic \
    nbclient \
    nbconvert \
    nbformat \
    nest-asyncio \
    netpyne \
    NEURON \
    notebook \
    notebook_shim \
    numpy \
    packaging \
    pandas \
    pandocfilters \
    parso \
    Pebble \
    pexpect \
    pickleshare \
    Pillow \
    platformdirs \
    prometheus-client \
    prompt-toolkit \
    psutil \
    ptyprocess \
    pure-eval \
    pycparser \
    Pygments \
    PyGObject \
    pyparsing \
    pyrsistent \
    python-dateutil \
    python-json-logger \
    pytz \
    PyYAML \
    pyzmq \
    qtconsole \
    QtPy \
    rfc3339-validator \
    rfc3986-validator \
    ruamel.yaml \
    ruamel.yaml.clib \
    schema \
    scipy \
    Send2Trash \
    six \
    sniffio \
    soupsieve \
    ssh-import-id \
    stack-data \
    terminado \
    tinycss2 \
    tornado \
    tqdm \
    traitlets \
    tzdata \
    uri-template \
    wcwidth \
    webcolors \
    webencodings \
    websocket-client \
    widgetsnbextension \
    xyzservices \
    inspyred

# Set necessary environment variables
ENV LD_LIBRARY_PATH=/opt/miniconda3/lib:$LD_LIBRARY_PATH

# Set necessary environment variables
ENV LD_LIBRARY_PATH=/opt/miniconda3/lib:$LD_LIBRARY_PATH

# Prepare a writable directory for Fontconfig cache
RUN mkdir /opt/fontconfig && \
    chmod 777 /opt/fontconfig

# Set environment variable for Fontconfig to use the new cache directory
ENV FONTCONFIG_PATH=/opt/fontconfig

WORKDIR /app

# Create a non-root user
RUN useradd -m myuser -d /opt/myuser

# Set HOME environment variable
ENV HOME=/opt/myuser

# Make the home directory writable
RUN chmod a+w /opt/myuser
RUN chmod -R o+w /opt/myuser \
    && fc-cache --really-force --verbose

# Switch to the non-root user
USER myuser

# Set the default command to run when the container starts
CMD ["python"]