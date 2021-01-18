# Run status

A complete MILO run can take several hours to complete which may even be much longer if  slower hardware or large datasets are being trained. While a run is completing, you will see a screen which shows all the pipeline combinations MILO’s engine is performing. A sample of this "Wheel of pipelines" is shown below (which can be magnified to see the details of the various individual ML pipelines within the wheel).

![Pipeline Combinations](./images/running.gif)

While a job is running you may also see all queued jobs by hitting the button in the top right corner which contains the three animated dots as shown below:

![Running List](./images/running.png)

To cancel a run, hit the red button containing the X in the image above and you will be presented with a confirmation to cancel your job:

![Cancel Run](./images/cancel-run.png)

Once the training is completed, MILO will automatically advance you to the "Results" page which will enable you to evaluate, fine tune or deploy the various models that meets your needs.
