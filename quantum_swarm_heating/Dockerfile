ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:latest
FROM $BUILD_FROM

# Install Python + pip + build tools
RUN apk add --no-cache \
        python3 \
        py3-pip \
        py3-numpy \
        build-base \
        python3-dev

# Install PyTorch CPU wheels (manylinux)
RUN pip3 install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install your other deps
RUN pip3 install --no-cache-dir networkx requests

COPY qsh_script.py /qsh_script.py
COPY run.sh /run.sh

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]