FROM ghcr.io/home-assistant/amd64-base-python

# Install dependencies
RUN pip install --no-cache-dir numpy==1.26.4 \
    && pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir networkx requests

# Copy script
COPY qsh_script.py /qsh_script.py
COPY run.sh /run.sh

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]