FROM rapidsai/rapidsai:0.17-cuda11.0-runtime-ubuntu18.04

SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

WORKDIR rapids/

RUN apt-get update && apt-get install -y --no-install-recommends eog cmake gcc python3-dev build-essential r-cran-mgcv

ARG CACHEBUST=1

RUN pip install git+https://github.com/LouisFaure/scFates && git clone https://github.com/LouisFaure/scFates && mv scFates/docs/notebooks/* . && rm -r scFates/

CMD ["/bin/bash"]

ENTRYPOINT ["/usr/bin/tini","--","/opt/docker/bin/entrypoint"]

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786
