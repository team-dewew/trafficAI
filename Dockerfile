FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /repo

# Install required system packages for OpenCV (headless should not need much, but just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /repo/
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . /repo/

# Run tests or wait for command
CMD ["bash"]
