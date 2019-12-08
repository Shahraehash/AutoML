import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AlertController, LoadingController } from '@ionic/angular';
import { parse } from 'papaparse';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-upload',
  templateUrl: 'upload.html',
  styleUrls: ['upload.scss'],
})
export class UploadPage implements OnInit {
  @Input() stepFinished;
  labels = [];
  uploadForm: FormGroup;

  constructor(
    public backend: BackendService,
    private alertController: AlertController,
    private formBuilder: FormBuilder,
    private loadingController: LoadingController
  ) {
    this.uploadForm = this.formBuilder.group({
      label_column: ['', Validators.required],
      train: ['', Validators.required],
      test: ['', Validators.required]
    });
  }

  ngOnInit() {
    this.backend.updatePreviousJobs();
  }

  onSubmit() {
    const formData = new FormData();
    formData.append('train', this.uploadForm.get('train').value);
    formData.append('test', this.uploadForm.get('test').value);
    formData.append('label_column', this.uploadForm.get('label_column').value);

    this.backend.submitData(formData).subscribe(
      () => {
        this.stepFinished('upload', this.labels.length);
      },
      async () => {
        const alert = await this.alertController.create({
          header: 'Unable to Upload Data',
          message: 'Please make sure the backend is reachable and try again.',
          buttons: ['Dismiss']
        });

        await alert.present();
      }
    );

    return false;
  }

  onFileSelect(event) {
    if (event.target.files.length === 1) {
      const file = event.target.files[0];

      if (event.target.name === 'train') {
        parse(file, {
          complete: reply => this.labels = reply.data[0]
        });
      }

      this.uploadForm.get(event.target.name).setValue(file);
    }
  }

  async trainPrior(id) {
    const loading = await this.loadingController.create({
      message: 'Creating New Job'
    });

    await loading.present();
    await this.backend.cloneJob(id).toPromise();
    await loading.dismiss();
    this.stepFinished('upload');
  }

  viewPrior(id) {
    this.backend.currentJobId = id;
    this.stepFinished('upload');
    this.stepFinished('train');
  }
}
