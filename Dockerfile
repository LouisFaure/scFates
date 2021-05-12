FROM rapidsai/rapidsai:0.19-cuda11.0-runtime-ubuntu20.04

SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

WORKDIR rapids/

RUN apt-get update && apt-get install -y --no-install-recommends eog cmake gcc python3-dev build-essential r-cran-mgcv

ARG CACHEBUST=1

RUN pip install git+https://github.com/j-bac/elpigraph-python.git
RUN pip install scFates --ignore-installed
RUN git clone https://github.com/LouisFaure/scFates_notebooks

CMD ["/bin/bash"]

ENTRYPOINT ["/usr/bin/tini","--","/opt/docker/bin/entrypoint"]

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786
