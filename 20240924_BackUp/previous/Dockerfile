# Tensorflow GPU Jupyter Base
FROM tensorflow/tensorflow:2.8.4-gpu-jupyter


# install LaTeX
RUN apt-get update -y && apt-get install wget texlive texlive-latex-extra cm-super texlive-fonts-recommended dvipng -y

RUN apt-get update -y && apt-get autoremove -y && apt-get autoclean 


# install packages into the default python release
RUN python -m pip install --upgrade pip setuptools importlib

RUN python -m pip install --force-reinstall jupyter notebook jupyterlab latex pandas matplotlib scikit-learn scipy numpy==1.22.* hist mplhep numba
    
RUN python -m pip install energyflow


# download madgraph
RUN mkdir /tf/madgraph && cd /tf/madgraph && wget https://launchpad.net/mg5amcnlo/lts/2.9.x/+download/MG5_aMC_v2.9.16.tar.gz && tar -xzpvf MG5_aMC_v2.9.16.tar.gz

ENV PATH="$PATH:/tf/madgraph/MG5_aMC_v2_9_16"


# jupyter config
RUN jupyter lab --generate-config

RUN echo "c.NotebookApp.allow_origin = '*'">>/root/.jupyter/jupyter_notebook_config.py

RUN echo "c.NotebookApp.ip = '0.0.0.0'">>/root/.jupyter/jupyter_notebook_config.py

# set password for jupyter server, if wanted. Needs to be sha256 hashed
# RUN echo "c.NotebookApp.password='sha256:9a6c6b2d8a54:4df24f914608980192a3d3580de6cef219222fc8cf2c7d5370db690db2883cb4'">>/root/.jupyter/jupyter_notebook_config.py


# set the entrypoint
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0", "--port=8888", "--allow-root"]
