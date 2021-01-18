# set base image (host OS)
FROM python:3.7

# Expose ports
EXPOSE 5000

# install OS dependencies
RUN apt-get update
RUN apt-get -y install curl gnupg sudo
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get -y install nodejs rabbitmq-server

# create user
RUN useradd -r -m -g sudo milo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER milo

# update path for Python
ENV PATH="/home/milo/.local/bin:${PATH}"

# set the working directory in the container
WORKDIR /milo

# copy the dependencies file to the working directory
COPY --chown=milo package.json .
COPY --chown=milo package-lock.json .
COPY --chown=milo requirements.txt .
COPY --chown=milo server.py .
COPY --chown=milo worker.py .
COPY --chown=milo static/ static/
COPY --chown=milo ml/ ml/
COPY --chown=milo common/ common/
COPY --chown=milo api/ api/

# create data directory
RUN mkdir data

# install app dependencies
RUN npm install --ignore-scripts
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "npm", "run", "run-docker" ]
