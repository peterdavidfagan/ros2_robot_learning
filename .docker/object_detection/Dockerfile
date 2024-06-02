ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base
MAINTAINER Peter David Fagan "peterdavidfagan@gmail.com"

# set default DDS to Cyclone
ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION rmw_cyclonedds_cpp

# install system dependencies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
        make \
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
        git \
        neovim \
        libxcb-cursor0 \ 
        libgmp-dev && \
        rm -rf /var/lib/apt/lists/*

# set up ROS workspace
ENV ROS_UNDERLAY /root/object_detection_ws/install
WORKDIR $ROS_UNDERLAY/..
COPY src/object_detection ./src/object_detection
COPY src/applications/transporter_deployment ./src/applications/transporter_deployment
COPY pyproject.toml .

# install python virtual environment
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
RUN /pyenv/bin/pyenv install 3.10.6
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.10.6 && pip install numpy poetry setuptools wheel six auditwheel
RUN mkdir -p .venv
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.10.6 && poetry config virtualenvs.in-project true --local && poetry install --only camera,huggingface

# set up workspace           
RUN . .venv/bin/activate && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    apt-get update && \
    apt-get install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp python-empy && \
    pip uninstall em && \
    rosdep update && \
    DEBIAN_FRONTEND=noninteractive \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} && \
    colcon build && \
    rm -rf /var/lib/apt/lists/*

# create entrypoint
COPY .docker/ros_entrypoint.sh /sbin/ros_entrypoint.sh
RUN sudo chmod 755 /sbin/ros_entrypoint.sh
ENTRYPOINT ["/sbin/ros_entrypoint.sh"]
