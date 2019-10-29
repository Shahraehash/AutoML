import { Component, Input } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AlertController } from '@ionic/angular';
import { parse } from 'papaparse';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-upload',
  templateUrl: 'upload.html',
  styleUrls: ['upload.scss'],
})
export class UploadPage {
  @Input() stepFinished;
  labels = [];
  previousJobForm: FormGroup;
  uploadForm: FormGroup;

  constructor(
    public backend: BackendService,
    private alertController: AlertController,
    private formBuilder: FormBuilder
  ) {
    this.previousJobForm = this.formBuilder.group({
      previousJob: ['', Validators.required]
    });

    this.uploadForm = this.formBuilder.group({
      label_column: ['', Validators.required],
      train: ['', Validators.required],
      test: ['', Validators.required]
    });
  }

  onSubmit() {
    const previous = this.previousJobForm.get('previousJob').value;

    if (previous) {
      this.backend.currentJobId = previous;
      this.stepFinished('upload');
      return;
    }

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
}
