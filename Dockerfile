# The builder image
FROM python:3.9-slim as builder

WORKDIR /usr/src/app

# Update the system and install pip
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip git

# Copy your requirements file (if you have one)
COPY requirements.txt ./

# Install Python dependencies
RUN pip3 install --upgrade pip &&\
    pip3 install --no-cache-dir -r requirements.txt

# The base image
FROM python:3.9-slim

WORKDIR /usr/src/app

# Copy everything from the builder image
COPY --from=builder /usr/src/app .

# Copy your Python script
COPY *.py .

# Run your Python script
CMD ["python3", "dummy.py"]