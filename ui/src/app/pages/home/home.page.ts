import { Component, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { STEPPER_GLOBAL_OPTIONS } from '@angular/cdk/stepper';
import { MatStepper } from '@angular/material';
import { PopoverController } from '@ionic/angular';
import { Observable, timer, of } from 'rxjs';
import { filter, switchMap, catchError } from 'rxjs/operators';

import { PendingTasksComponent } from '../../components/pending-tasks/pending-tasks.component';
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
  @ViewChild('stepper', {static: false}) stepper: MatStepper;

  pendingTasks$: Observable<PendingTasks>;
  pauseUpdates = false;
  uploadForm: FormGroup;
  trainForm: FormGroup;
  featureCount: number;

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
    this.pendingTasks$ = timer(0, 5000).pipe(
      filter(() => !this.pauseUpdates),
      switchMap(() => this.backend.getPendingTasks().pipe(
        catchError(() => of({active: [], scheduled: []}))
      ))
    );
  }

  exportCSV() {
    window.open(this.backend.exportCSV(), '_self');
  }

  reset() {
    this.stepper.reset();
    this.uploadForm.reset();
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

  stepFinished(event) {
    switch (event.state) {
      case 'upload':
        this.featureCount = event.data;
        this.uploadForm.get('upload').setValue('true');
        break;
      case 'train':
        this.trainForm.get('train').setValue('true');
    }

    this.stepper.next();
  }
}
