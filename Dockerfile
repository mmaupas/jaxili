FROM condaforge/miniforge3

LABEL Description="JaxILI Docker Image with Python 3.12"
WORKDIR /home
ENV SHELL /bin/bash

ENV ENV_NAME=jaxili

RUN apt-get update
RUN apt-get install build-essential -y

COPY conda_env.yml .

RUN conda env create -f conda_env.yml

ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

CMD ["bash"]