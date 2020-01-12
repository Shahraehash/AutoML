import { Component, AfterViewInit, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { STEPPER_GLOBAL_OPTIONS } from '@angular/cdk/stepper';
import { MatStepper } from '@angular/material';
import { ActivatedRoute } from '@angular/router';
import { PopoverController } from '@ionic/angular';
import { Observable, timer, of } from 'rxjs';
import { filter, switchMap, catchError } from 'rxjs/operators';

import { PendingTasksComponent } from '../../components/pending-tasks/pending-tasks.component';
import { TrainComponent } from '../../components/train/train.component';
import { BackendService } from '../../services/backend.service';
import { PendingTasks } from '../../interfaces';

@Component({
  selector: 'app-search',
  templateUrl: './search.page.html',
  styleUrls: ['./search.page.scss'],
  providers: [{
    provide: STEPPER_GLOBAL_OPTIONS, useValue: {showError: true}
  }]
})
export class SearchPage implements OnInit, AfterViewInit {
  @ViewChild('stepper', {static: false}) stepper: MatStepper;
  @ViewChild('train', {static: false}) train: TrainComponent;

  pendingTasks$: Observable<PendingTasks>;
  pauseUpdates = false;
  uploadForm: FormGroup;
  trainForm: FormGroup;
  featureCount: number;

  constructor(
    private activatedRoute: ActivatedRoute,
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

  ngAfterViewInit() {
    const trainId = this.activatedRoute.snapshot.params.trainId;
    if (trainId) {
      this.backend.currentJobId = trainId;
      this.stepFinished({state: 'upload'});
    }

    const statusId = this.activatedRoute.snapshot.params.statusId;
    const taskId = this.activatedRoute.snapshot.params.taskId;
    if (statusId && taskId) {
      this.backend.currentJobId = statusId;
      this.stepFinished({state: 'upload'});
      window.history.pushState('', '', `/search/status/${statusId}/${taskId}`);
      setTimeout(() => this.train.startMonitor(taskId), 1);
    }

    const resultId = this.activatedRoute.snapshot.params.resultId;
    if (resultId) {
      this.backend.currentJobId = resultId;
      this.stepFinished({state: 'upload'});
      this.stepFinished({state: 'train'});
    }
  }

  exportCSV() {
    window.open(this.backend.exportCSV(), '_self');
  }

  reset() {
    this.stepper.reset();
    this.uploadForm.reset();
    window.history.pushState('', '', `/search`);
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
        window.history.pushState('', '', `/search/train/${this.backend.currentJobId}`);
        break;
      case 'train':
        this.trainForm.get('train').setValue('true');
        window.history.pushState('', '', `/search/result/${this.backend.currentJobId}`);
    }

    this.stepper.next();
  }
}
