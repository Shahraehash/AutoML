# set base image (host OS)
FROM python:3.7

# expose ports
EXPOSE 5000
EXPOSE 8443

# install OS dependencies
RUN apt-get update
RUN apt-get -y install curl gnupg sudo libssl-dev openssl
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get -y install nodejs rabbitmq-server

# create user
RUN useradd -r -m -g sudo milo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER milo

# update path for Python
ENV PATH="/home/milo/.local/bin:${PATH}"

# set the working directory in the container
RUN sudo mkdir /milo
RUN sudo chown -R milo /milo
WORKDIR /milo

# create data directory
RUN mkdir data
RUN mkdir ssl

# copy the Python dependencies to the working directory
COPY requirements.txt .

# install Python requirements
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# copy the dependencies file to the working directory
COPY package.json .
COPY package-lock.json .

# install app dependencies
RUN npm install --ignore-scripts

# copy remaining Python code
COPY server.py .
COPY worker.py .
COPY ml/ ml/
COPY common/ common/
COPY api/ api/
COPY uwsgi.ini .
COPY preprocessor/modules/ preprocessor/modules/

# copy static assets (UI and documentation)
COPY static/ static/

# copy client assets
COPY client/ client/

# generate SSL certificate
RUN openssl req -x509 -nodes \
    -subj  "/CN=localhost" \
    -addext "subjectAltName = DNS:localhost" \
    -addext "keyUsage = digitalSignature" \
    -addext "extendedKeyUsage = serverAuth" \
    -newkey rsa:2048 -keyout ssl/milo.key \
    -out ssl/milo.crt

# if present, bundle the educational license
COPY *license.pub data/
COPY *licensefile.skm data/

# start the application
CMD [ "npm", "run", "run-docker" ]
