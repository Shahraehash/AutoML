import { Component, AfterViewInit, OnInit, ViewChild } from '@angular/core';
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
  featureCount: number;

  constructor(
    public activatedRoute: ActivatedRoute,
    public backend: BackendService,
    private popoverController: PopoverController
  ) {}

  ngOnInit() {
    this.pendingTasks$ = timer(0, 5000).pipe(
      filter(() => !this.pauseUpdates),
      switchMap(() => this.backend.getPendingTasks().pipe(
        catchError(() => of({active: [], scheduled: []}))
      ))
    );
  }

  ngAfterViewInit() {
    this.backend.currentDatasetId = this.activatedRoute.snapshot.params.dataId;
    this.backend.currentJobId = this.activatedRoute.snapshot.params.jobId;
    this.stepFinished({nextStep: this.activatedRoute.snapshot.params.step});
    const taskId = this.activatedRoute.snapshot.params.taskId;
    if (taskId) {
      setTimeout(() => this.train.startMonitor(taskId), 1);
    }

    this.stepper.selectionChange.subscribe(event => {
      switch (event.selectedIndex) {
        case 3:
          window.history.pushState('', '', `/search/${this.backend.currentDatasetId}/job/${this.backend.currentJobId}/result`);
          break;
        case 2:
          window.history.pushState('', '', `/search/${this.backend.currentDatasetId}/job/${this.backend.currentJobId}/train`);
          break;
        case 1:
          window.history.pushState('', '', `/search/${this.backend.currentDatasetId}/explore`);
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
    this.backend.currentJobId = undefined;
    this.backend.currentDatasetId = undefined;
    this.stepper.reset();
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
    switch (event.nextStep) {
      case 'explore':
        setTimeout(() => this.stepper.selectedIndex = 1, 1);
        break;
      case 'train':
        setTimeout(() => this.stepper.selectedIndex = 2, 1);
        break;
      case 'result':
        setTimeout(() => this.stepper.selectedIndex = 3, 1);
    }
  }
}
