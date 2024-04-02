FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Add user and update
RUN /bin/bash -c 'apt update; \
    apt install -y software-properties-common; \
    add-apt-repository ppa:deadsnakes/ppa; \
    apt update; \
    apt install -y python3; \
    apt install -y python3-pip; \
    python3 -m pip install --upgrade pip; \
    useradd -m dev'
#-------------------------------------------------

# Prompt formatting
RUN /bin/bash -c 'apt install -y vim'
RUN echo "parse_git_branch() {\n git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\\\1)/' \n}" >> /home/dev/.bashrc
RUN echo "export PS1=\"\[\e[34m\]\u@\[\e[34m\]\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ \" " >> /home/dev/.bashrc
#-------------------------------------------------

# Install python packages
COPY requirements.txt /requirements.txt
RUN /bin/bash -c 'pip3 install -r /requirements.txt'
#-------------------------------------------------

# Install git
RUN /bin/bash -c 'DEBIAN_FRONTEND=noninteractive apt install -y git-all'
#-------------------------------------------------

# USER dev