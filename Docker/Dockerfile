# VERSION: 0.1.0
# DESCRIPTION: Basic extensible Jupyter Notebook/Lab Container
# BUILD: docker build --rm -t docker-jupyter-extensible .

FROM jupyter/minimal-notebook:ubuntu-20.04
USER root
RUN apt-get update && apt-get install -y \
    libfcgi-dev                          \
	wget								 \
	curl								 \
	git								     \
    libglib2.0-dev                       \
    libsoup2.4-dev                       \
    libssl-dev                           \
    python3-pip                          \
	ffmpeg	                             \
    unzip                                \
    valac                                \
	bash								\
    && rm -rf /var/lib/apt/lists/* &&	 \
    apt-get update 				 

#RUN git clone https://github.com/stylegan-human/StyleGAN-Human.git

ADD https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip /tmp
RUN unzip /tmp/ninja-linux.zip -d /usr/local/bin
RUN rm -f /tmp/ninja-linux.zip
RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 

USER $NB_UID
RUN conda update conda
COPY environment.yml environment.yml
RUN conda env create -f environment.yml 

