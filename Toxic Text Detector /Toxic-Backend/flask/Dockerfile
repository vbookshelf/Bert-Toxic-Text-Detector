# Use the Python3.7.2 container image
FROM python:3.7.2-stretch

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install the dependencies
# These commands are run inside the docker container.
# Docker builds a seperate image each time it sees a RUN command.
# Therefore use the && command after the first RUN.
# Video 30

RUN pip3 install torch torchvision --no-cache-dir

RUN pip install tensorflow==2.1.0 --no-cache-dir

RUN pip install transformers==2.7.0 --no-cache-dir

RUN pip install -r requirements.txt

# run the command to start uWSGI
CMD ["uwsgi", "app.ini"]