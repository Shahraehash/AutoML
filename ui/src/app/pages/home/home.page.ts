import { Component, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatStepper } from '@angular/material';
import {STEPPER_GLOBAL_OPTIONS} from '@angular/cdk/stepper';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
  providers: [{
    provide: STEPPER_GLOBAL_OPTIONS, useValue: {showError: true}
  }]
})
export class HomePage {
  @ViewChild('stepper', {static: false}) stepper: MatStepper;

  uploadForm: FormGroup;
  trainForm: FormGroup;
  featureCount: number;

  stepFinished = (step, extra) => {
    this.stepper.next();

    switch (step) {
      case 'upload':
        this.featureCount = extra;
        this.uploadForm.get('upload').setValue('true');
        break;
      case 'train':
        this.trainForm.get('train').setValue('true');
    }
  }

  constructor(
    private formBuilder: FormBuilder
  ) {
    this.uploadForm = this.formBuilder.group({
      upload: ['', Validators.required]
    });

    this.trainForm = this.formBuilder.group({
      train: ['', Validators.required]
    });
  }
}
