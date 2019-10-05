import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AlertController } from '@ionic/angular';
import { parse } from 'papaparse';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {
  labels = [];
  uploadForm: FormGroup;

  constructor(
    private alertController: AlertController,
    private backend: BackendService,
    private formBuilder: FormBuilder,
    private router: Router
  ) {
    this.uploadForm = this.formBuilder.group({
      label_column: ['', Validators.required],
      train: ['', Validators.required],
      test: ['', Validators.required]
    });
  }

  onSubmit() {
    const formData = new FormData();
    formData.append('train', this.uploadForm.get('train').value);
    formData.append('test', this.uploadForm.get('test').value);
    formData.append('label_column', this.uploadForm.get('label_column').value);

    this.backend.submitData(formData).subscribe(
      () => this.router.navigate(['/train', {upload: true}]),
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

      if (!this.labels.length) {
        parse(file, {
          complete: reply => this.labels = reply.data[0]
        });
      }

      this.uploadForm.get(event.target.name).setValue(file);
    }
  }
}
