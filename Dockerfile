FROM python:3.9-slim-buster
RUN apt-get update -y && apt-get install graphviz graphviz-dev -y
RUN pip install --upgrade pip
COPY requirements_runtime.txt /tmp
COPY requirements_test.txt /tmp
WORKDIR /tmp
# need to install torch separately with no cache option as large package -otherwise process gets killed due to excessive memory consumption
RUN pip --no-cache-dir install torch \
    && pip install -r requirements_runtime.txt \
    && pip install -r requirements_test.txt
COPY . /bayesian
WORKDIR /bayesian
ENTRYPOINT ["/bin/bash"]


# docker build -t bayesian .
# docker run -it bayesian
