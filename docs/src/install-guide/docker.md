# Docker

MILO's docker image hosts everything needed to get up and running with MILO.

## 1) Install Docker Desktop

Download [Docker Desktop](https://www.docker.com/products/docker-desktop) and follow the instructions from the installer to get it installed on your system.

## 2) Open Docker Dashboard and Configuring Docker

This is a one time step after installing Docker to ensure enough resources are allocated for MILO. This is done using the Docker Desktop Dashboard launched as shown here:

![Docker Dashboard Link](./images/docker-dashboard-link.png)

Inside the dashboard click the button with a gear icon as shown:

![Docker Settings Button](./images/docker-settings-button.png)

This will navigate to a settings page which has sections on the left hand side. Please navigate to the `Resources` section as shown below:

![Docker Resources](./images/docker-resources.png)

Here, please adjust the settings to maximize the available CPU and memory to meet at least 4 CPUs and at least 8GB of RAM.

## 3) Download MILO Image

Please contact us to get a download link to our docker image. The file downloaded by the link provided will be a TAR file which should not need to be extracted.

## 4) Import MILO Image

Docker Desktop does not currently provide a way to import an image but fortunately also installs Docker CLI. In this section, we will use Docker CLI
to import the MILO docker image.

Please open any terminal and run the following command:

```sh
# Replace the file name with the one you downloaded

docker load -i milo-1.0.0.tar
```

## 5) First Time Starting MILO

Once the MILO image is loaded, the remainder of the deployment can be performed within Docker Desktop.

To start this image first open the Docker Desktop Dashboard and navigate to the `Images` section:

![Docker Images](./images/docker-images.png)

You will see an image named `milo`. If you hover over this two buttons will appear one of which is labeled `Run` as shown here:

![Docker Run Image](./images/docker-image-run.png)

After clicking this button you will be presented with a modal with some optional settings. Please expand this section
and configure it as shown in the image below:

![Docker Run Settings](./images/docker-run-settings.png)

The port 5000 is HTTP application port and is being forwarded to the host machine's port 5000 in the above image. The data
directory is where MILO places input data and results and can be mapped to the local machine for consistency between runs.

::: warning
License keys are associated to the hardware and therefore will become invalidated as the hardware changes. Keep in mind, each time
the `Run` button is clicked Docker creates new virtual hardware.
:::

## 6) Start Using MILO

After you start MILO, you should see some output similar to the following:

![Docker Running](./images/docker-running.png)

This informs you that MILO is ready to start processing requests and you can now launch your browser and navigating
to the following URL: <http://localhost:5000>.

## Stopping MILO

You can always stop and re-start MILO using the Dashboard by clicking the `Stop` button as shown below:

![Docker Stop](./images/docker-stop.png)

::: warning
Stopping MILO will interrupt any training currently in progress so ensure no tasks are pending before stopping.
:::

Once stopped, the button will change to a play button allowing you to start MILO again.
