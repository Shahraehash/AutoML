# set base image (host OS)
FROM python:3.7

# expose ports
EXPOSE 5000
EXPOSE 8443

# install OS dependencies
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get update
RUN apt-get -y install curl gnupg sudo libssl-dev openssl nodejs rabbitmq-server

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
COPY --chown=milo:sudo requirements.txt .

# install Python requirements
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# copy the dependencies file to the working directory
COPY --chown=milo:sudo package.json .
COPY --chown=milo:sudo package-lock.json .

# install app dependencies
RUN npm install --ignore-scripts

# copy remaining Python code
COPY --chown=milo:sudo server.py .
COPY --chown=milo:sudo worker.py .
COPY --chown=milo:sudo ml/ ml/
COPY --chown=milo:sudo common/ common/
COPY --chown=milo:sudo api/ api/
COPY --chown=milo:sudo uwsgi.ini .
COPY --chown=milo:sudo preprocessor/modules/ preprocessor/modules/

# copy static assets (UI and documentation)
COPY --chown=milo:sudo static/ static/

# copy client assets
COPY --chown=milo:sudo client/ client/

# generate SSL certificate
RUN openssl req -x509 -nodes \
    -subj  "/CN=localhost" \
    -addext "subjectAltName = DNS:localhost" \
    -addext "keyUsage = digitalSignature" \
    -addext "extendedKeyUsage = serverAuth" \
    -newkey rsa:2048 -keyout ssl/milo.key \
    -out ssl/milo.crt

# if present, bundle the educational license
COPY --chown=milo:sudo *license.pub data/
COPY --chown=milo:sudo *licensefile.skm data/

# start the application
CMD [ "npm", "run", "run-docker" ]
