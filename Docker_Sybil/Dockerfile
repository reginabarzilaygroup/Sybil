### Derive from the base container image
FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.9-pt1.11-v1

### Install python packages
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

#### Configure the entry point
USER root
COPY run.sh /root/
ENTRYPOINT ["/bin/bash", "/root/run.sh"]