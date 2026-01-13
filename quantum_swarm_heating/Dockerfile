FROM homeassistant/amd64-base-python:3.12

RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir networkx requests

COPY qsh_script.py /qsh_script.py

CMD ["python3", "/qsh_script.py"]
