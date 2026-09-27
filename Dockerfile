# GPU image with Python 3.11 / 3.12 + CUDA 12.4 (matches torch==2.6.0 in requirements.txt).
#   bash weights/download.sh          # once, with internet (or copy the .pt files and InternVL2_5-1B/ into weights/)
#   docker build -t team .
#   docker run --gpus all --network none -v /data/test:/data/test team \
#       python run_submission.py --videos /data/test --out predictions.json
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /repo
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /repo/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /repo/
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/repo/predictions.json"]

