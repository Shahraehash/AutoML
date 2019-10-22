import { Component, OnInit, ViewChild } from '@angular/core';
import { MatStepper } from '@angular/material';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
})
export class HomePage implements OnInit {
  @ViewChild('stepper', {static: false}) stepper: MatStepper;

  featureCount;
  uploadCompleted = false;

  stepFinished = (step, extra) => {
    this.stepper.next();

    switch (step) {
      case 'upload':
        this.uploadCompleted = true;
        this.featureCount = extra;
    }
  }

  constructor() {}

  ngOnInit() {}
}
