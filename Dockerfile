FROM debian:bookworm

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    wget \
    ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install python=3.7

RUN /opt/conda/bin/conda install nodejs

RUN /opt/conda/bin/conda upgrade -n base conda

RUN /opt/conda/bin/conda install conda-forge::swig

# Copy the requirements text into the container at /
ADD /requirements.txt /

# install dependencies
RUN pip install -r requirements.txt

# Create a directory to mount homework at run 
RUN mkdir hw2

# Set the working directory to /hw2
WORKDIR /hw2

EXPOSE 8889

# DEBUG: Check lspci output
RUN apt-get update
RUN apt-get install pciutils -y

# Install the necessary package for add-apt-repository
RUN apt-get update && \
    apt-get install software-properties-common -y

# Install CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-debian12-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    dpkg -i cuda-repo-debian12-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    cp /var/cuda-repo-debian12-12-4-local/cuda-C5AA6424-keyring.gpg /usr/share/keyrings/ && \
    add-apt-repository contrib && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4 --silent

# Install the package and run ipython in no-browser mode
CMD ["sh", "-c", "pip install -e . && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port=8889"]