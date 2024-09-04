# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set the working directory inside the container
WORKDIR /app

# Install additional packages and clean up to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgl1-mesa-glx \
    && pip install \
    pydot \
    tensorflow-datasets \
    tensorflow_probability \
    opencv-python \
    matplotlib \
    seaborn \
    scikit-learn \
    albumentations \  
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
