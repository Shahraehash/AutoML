import { Component, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { STEPPER_GLOBAL_OPTIONS } from '@angular/cdk/stepper';
import { MatStepper } from '@angular/material';
import { PopoverController } from '@ionic/angular';
import { Observable, timer } from 'rxjs';
import { filter, switchMap } from 'rxjs/operators';

import { PendingTasksComponent } from '../../components/pending-tasks/pending-tasks.component';
import { TrainPage } from '../../components/train/train.page';
import { BackendService } from '../../services/backend.service';
import { PendingTasks } from '../../interfaces';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
  providers: [{
    provide: STEPPER_GLOBAL_OPTIONS, useValue: {showError: true}
  }]
})
export class HomePage implements OnInit {
  @ViewChild('train', {static: false}) train: TrainPage;
  @ViewChild('stepper', {static: false}) stepper: MatStepper;

  pendingTasks$: Observable<PendingTasks>;
  pauseUpdates = false;
  uploadForm: FormGroup;
  trainForm: FormGroup;
  featureCount: number;

  stepFinished = (step, extra) => {
    switch (step) {
      case 'upload':
        this.featureCount = extra;
        this.train.training = false;
        this.uploadForm.get('upload').setValue('true');
        break;
      case 'train':
        this.trainForm.get('train').setValue('true');
    }

    this.stepper.next();
  }

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder,
    private popoverController: PopoverController
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
      filter(() => !this.pauseUpdates),
      switchMap(() => this.backend.getPendingTasks())
    );
  }

  exportCSV() {
    window.open(this.backend.exportCSV(), '_self');
  }

  async openPendingTasks(event, pendingTasks) {
    this.pauseUpdates = true;
    const popover = await this.popoverController.create({
      cssClass: 'wide-popover',
      component: PendingTasksComponent,
      componentProps: {firstViewData: pendingTasks},
      event,
      translucent: true
    });

    await popover.present();
    await popover.onDidDismiss();
    this.pauseUpdates = false;
  }
}
