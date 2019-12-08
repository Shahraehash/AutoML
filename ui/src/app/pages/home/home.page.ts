import { Component, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { STEPPER_GLOBAL_OPTIONS } from '@angular/cdk/stepper';
import { MatStepper } from '@angular/material';
import { Observable, timer } from 'rxjs';
import { switchMap } from 'rxjs/operators';

import { BackendService } from '../../services/backend.service';
import { TaskStatus } from '../../interfaces';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
  providers: [{
    provide: STEPPER_GLOBAL_OPTIONS, useValue: {showError: true}
  }]
})
export class HomePage implements OnInit {
  @ViewChild('stepper', {static: false}) stepper: MatStepper;

  pendingTasks$: Observable<TaskStatus[]>;
  uploadForm: FormGroup;
  trainForm: FormGroup;
  featureCount: number;

  stepFinished = (step, extra) => {
    switch (step) {
      case 'upload':
        this.featureCount = extra;
        this.uploadForm.get('upload').setValue('true');
        break;
      case 'train':
        this.trainForm.get('train').setValue('true');
    }

    this.stepper.next();
  }

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder
  ) {
    this.uploadForm = this.formBuilder.group({
      upload: ['', Validators.required]
    });

    this.trainForm = this.formBuilder.group({
      train: ['', Validators.required]
    });
  }

  ngOnInit() {
    this.pendingTasks$ = timer(0, 10000).pipe(
      switchMap(() => this.backend.getPendingTasks())
    );
  }

  exportCSV() {
    window.open(this.backend.exportCSV(), '_self');
  }

  openPendingTasks() {

  }
}
