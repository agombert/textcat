# Use latest Python runtime as a parent image
FROM python:3.6.9-slim-buster

# Meta-data
LABEL maintainer="Arnault Gombert <arnault.gombert@gmail.com>" \
      description="Docker To train to detect hate speech"

# Set the working directory to /app
WORKDIR /classifiers/

# Copy the current directory contents into the container at /app
COPY . /classifiers/

# pip install
RUN pip install pip --upgrade && pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 8888

# Create mountpoint
# VOLUME /classifiers/data/

# ENTRYPOINT allows us to specify the default executible
# ENTRYPOINT ["python"]
