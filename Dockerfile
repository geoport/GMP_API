FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /app

# Install TensorRT libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer7=7.2.1-1+cuda10.2 \
    libnvinfer-dev=7.2.1-1+cuda10.2 \
    libnvinfer-plugin7=7.2.1-1+cuda10.2

# Install other dependencies
COPY requirements.txt ./
RUN apt-get install -y --no-install-recommends apt-utils curl libgomp1
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src .

# Expose the port for the app
EXPOSE 9011
