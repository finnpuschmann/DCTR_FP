# Tensorflow GPU Jupyter Base
FROM tensorflow/tensorflow:2.8.4-gpu-jupyter


# install LaTeX
RUN apt-get update -y && apt-get install texlive texlive-latex-extra cm-super texlive-fonts-recommended dvipng -y


# install packages into the default python release
RUN python -m pip install --upgrade pip setuptools importlib && \
    python -m pip install --upgrade --force-reinstall latex pandas matplotlib scikit-learn scipy numpy==1.22.* hist mplhep && \
    python -m pip install energyflow


# download madgraph
RUN mkdir /tf/madgraph && cd /tf/madgraph && apt-get install wget && \
    wget https://launchpad.net/mg5amcnlo/lts/2.9.x/+download/MG5_aMC_v2.9.16.tar.gz && tar -xzpvf MG5_aMC_v2.9.16.tar.gz

ENV PATH="$PATH:/tf/madgraph/MG5_aMC_v2_9_16"


# set the entrypoint
ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--allow-root"]
