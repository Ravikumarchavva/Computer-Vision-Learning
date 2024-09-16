# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install additional system packages and clean up to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgl1-mesa-glx \
    && pip install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
