import { Component, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatStepper } from '@angular/material';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
})
export class HomePage {
  @ViewChild('stepper', {static: false}) stepper: MatStepper;

  progressForm: FormGroup;
  featureCount: number;

  stepFinished = (step, extra) => {
    this.stepper.next();

    switch (step) {
      case 'upload':
        this.featureCount = extra;
        this.progressForm.get('upload').setValue(true);
    }
  }

  constructor(
    private formBuilder: FormBuilder
  ) {
    this.progressForm = this.formBuilder.group({
      upload: ['', Validators.required],
      train: ['', Validators.required],
      result: ['', Validators.required]
    });
  }
}
