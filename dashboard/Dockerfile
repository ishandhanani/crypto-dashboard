FROM python:3.9-slim

# Copy over important files and set working directory
COPY requirements.txt /tmp/ 
COPY ./app /app
WORKDIR "/app"

# Create environment from requirements.txt
# Below is to fix an error couldnt not build wheels for sip which uses PIP517
RUN pip install --upgrade pip setuptools wheel
# The next 2 lines install gcc which is needed for Bottleneck 
RUN apt-get update
RUN apt-get -y install gcc
# These install the rest of the packages
RUN pip install -r /tmp/requirements.txt

# Test activation
RUN echo "Making sure pandas is installed"
RUN python -c "import pandas"

# Run when container starts
# dashboard:server is look for server in dashboard.py
# app.server in dashboard.py fixed Application not callable error
EXPOSE 8050/tcp
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "dashboard:server"]

#use docker run -p 127.0.0.1:8050:8050/tcp dashboard where -p is publish. First one is the host port and second is the container port







