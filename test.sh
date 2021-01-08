docker run -it --rm -v /home/lfaure/scFates/:/rapids/rapids/ louisfaure/scfates:env-ready /bin/bash -c "pip install . coverage mock && coverage run -m pytest scFates/tests/ && coverage report -m"
