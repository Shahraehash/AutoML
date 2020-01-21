import { Component, AfterViewInit, ElementRef, OnInit, ViewChild } from '@angular/core';
import { STEPPER_GLOBAL_OPTIONS } from '@angular/cdk/stepper';
import { AngularFireAuth } from '@angular/fire/auth';
import { MatStepper } from '@angular/material';
import { ActivatedRoute, Router } from '@angular/router';
import { PopoverController } from '@ionic/angular';
import { Observable, timer, of } from 'rxjs';
import { filter, switchMap, catchError } from 'rxjs/operators';

import { PendingTasksComponent } from '../../components/pending-tasks/pending-tasks.component';
import { TrainComponent } from '../../components/train/train.component';
import { MiloApiService } from '../../services/milo-api/milo-api.service';
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

  featureCount: number;
  pendingTasks$: Observable<PendingTasks>;
  pauseUpdates = false;
  trainCompleted = false;

  constructor(
    public activatedRoute: ActivatedRoute,
    public afAuth: AngularFireAuth,
    public api: MiloApiService,
    private element: ElementRef,
    private popoverController: PopoverController,
    private router: Router
  ) {}

  async ngOnInit() {
    const pending = await this.api.getPendingTasks();
    this.pendingTasks$ = timer(0, 5000).pipe(
      filter(() => !this.pauseUpdates),
      switchMap(() => pending.pipe(
        catchError(() => of({active: [], scheduled: []}))
      ))
    );
  }

  ngAfterViewInit() {
    this.api.currentDatasetId = this.activatedRoute.snapshot.params.dataId;
    this.api.currentJobId = this.activatedRoute.snapshot.params.jobId;
    this.stepFinished({nextStep: this.activatedRoute.snapshot.params.step});
    const taskId = this.activatedRoute.snapshot.params.taskId;
    if (taskId) {
      setTimeout(() => this.train.startMonitor(taskId), 1);
    }

    this.stepper.selectionChange.subscribe(event => {
      switch (event.selectedIndex) {
        case 3:
          window.history.pushState('', '', `/search/${this.api.currentDatasetId}/job/${this.api.currentJobId}/result`);
          break;
        case 2:
          window.history.pushState('', '', `/search/${this.api.currentDatasetId}/job/${this.api.currentJobId}/train`);
          break;
        case 1:
          window.history.pushState('', '', `/search/${this.api.currentDatasetId}/explore`);
          break;
        case 0:
        default:
          window.history.pushState('', '', `/search`);
      }
    });
  }

  exportCSV() {
    window.open(this.api.exportCSV(), '_self');
  }

  reset() {
    this.api.currentJobId = undefined;
    this.api.currentDatasetId = undefined;
    this.trainCompleted = false;
    this.stepper.reset();

    /** Fixes stepper reset not clearing done state */
    for (const match of this.element.nativeElement.querySelectorAll('.mat-step-icon-state-done')) {
      match.classList.remove('mat-step-icon-state-done');
    }

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
        this.featureCount = event.data;
        setTimeout(() => this.stepper.selectedIndex = 2, 1);
        break;
      case 'result':
        this.trainCompleted = true;
        setTimeout(() => this.stepper.selectedIndex = 3, 1);
    }
  }

  async signOut() {
    await this.afAuth.auth.signOut();
    this.router.navigateByUrl('/login');
  }
}
