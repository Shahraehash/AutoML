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
  @ViewChild('stepper') stepper: MatStepper;
  @ViewChild('train') train: TrainComponent;

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
    const exploreId = this.activatedRoute.snapshot.params.exploreId;
    if (exploreId) {
      this.backend.currentDatasetId = exploreId;
      this.stepFinished({state: 'upload'});
    }

    const trainId = this.activatedRoute.snapshot.params.trainId;
    if (trainId) {
      this.backend.currentJobId = trainId;
      this.stepFinished({state: 'explore'});
    }

    const statusId = this.activatedRoute.snapshot.params.statusId;
    const taskId = this.activatedRoute.snapshot.params.taskId;
    if (statusId && taskId) {
      this.backend.currentJobId = statusId;
      this.stepFinished({state: 'explore'});
      setTimeout(() => this.train.startMonitor(taskId), 1);
    }

    const resultId = this.activatedRoute.snapshot.params.resultId;
    if (resultId) {
      this.backend.currentJobId = resultId;
      this.stepFinished({state: 'train'});
    }

    this.stepper.selectionChange.subscribe(event => {
      switch (event.selectedIndex) {
        case 3:
          window.history.pushState('', '', `/search/result/${this.backend.currentJobId}`);
          break;
        case 2:
          window.history.pushState('', '', `/search/train/${this.backend.currentJobId}`);
          break;
        case 1:
          window.history.pushState('', '', `/search/explore/${this.backend.currentDatasetId}`);
          break;
        case 0:
        default:
          window.history.pushState('', '', `/search`);
      }
    });
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
        this.stepper.selectedIndex = 1;
        break;
      case 'explore':
        this.uploadForm.get('upload').setValue('true');
        this.stepper.selectedIndex = 2;
        break;
      case 'train':
        this.uploadForm.get('upload').setValue('true');
        this.trainForm.get('train').setValue('true');
        this.stepper.selectedIndex = 2;
        setTimeout(() => this.stepper.selectedIndex = 3, 1);
    }
  }
}
