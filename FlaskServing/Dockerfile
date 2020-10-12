FROM amd64/python:3.7.6-slim-stretch

RUN apt update --fix-missing

RUN apt install -y htop
RUN apt install -y git
RUN apt install -y supervisor
RUN apt install -y libblas-dev liblapack-dev
RUN apt install -y libpython3-dev build-essential libpcre3-dev libatlas-dev
RUN apt install -y libhdf5-dev
RUN apt install -y libssl-dev 
RUN apt install -f libffi-dev

# set local datetimezone
ENV TZ Asia/Jakarta
RUN apt install -y locales

RUN sed -i -e 's/# id_ID.UTF-8 UTF-8/id_ID.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
RUN locale-gen id_ID.utf8

#create user
ARG user=tagger
ARG group=tagger
ARG uid=1000
ARG gid=1001
RUN adduser ${user}

USER ${user}
RUN mkdir /home/${user}/src
RUN mkdir /home/${user}/log

# add requirements files 
ADD requirements.txt /home/${user}/src/requirements.txt

USER root
RUN pip install -r /home/${user}/src/requirements.txt

EXPOSE 9901