import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import { AlertController, ModalController, LoadingController } from '@ionic/angular';
import { of, ReplaySubject } from 'rxjs';
import { delay, repeat, tap, takeUntil, catchError } from 'rxjs/operators';

import { MiloApiService } from '../../services/milo-api/milo-api.service';
import { PendingTasks } from '../../interfaces';
import { TrainComponent } from '../train/train.component';

@Component({
  selector: 'app-pending-tasks',
  templateUrl: './pending-tasks.component.html',
  styleUrls: ['./pending-tasks.component.scss'],
})
export class PendingTasksComponent implements OnInit, OnDestroy {
  @Input() firstViewData: PendingTasks;
  destroy$: ReplaySubject<boolean> = new ReplaySubject<boolean>();
  pendingTasks: PendingTasks;

  constructor(
    private alertController: AlertController,
    private api: MiloApiService,
    private loadingController: LoadingController,
    private modalController: ModalController
  ) {}

  async ngOnInit() {
    (await this.api.getPendingTasks()).pipe(
      catchError(_ => of(false)),
      takeUntil(this.destroy$),
      tap(pending => {
        if (typeof pending === 'boolean') {
          return;
        }

        if (!this.pendingTasks) {
          this.pendingTasks = pending;
        } else {
          this.pendingTasks.scheduled = pending.scheduled;
          
          /** Filter the tasks which are no longer active */
          this.pendingTasks.active = this.pendingTasks.active.filter(item => pending.active.find(q => q.id === item.id))

          pending.active.forEach(item => {

            /**
             * If the item has no progress information, skip the update. This is done
             * to maintain the currently displayed percentage rather then reverting to
             * a zero percent status.
             */
            if (item.state === 'PENDING' && item.current === 0 && item.total === 1) {
              return;
            }

            /** Find the current index of the matching item */
            const currentIndex = this.pendingTasks.active.findIndex(task => task.id === item.id);
            if (currentIndex === -1) {

              /** If a match is not found, this is a new item and should be added to the list */
              this.pendingTasks.active.push(item);
            } else {
              this.pendingTasks.active[currentIndex] = item;
            }
          });
        }
      }),
      delay(60000),
      repeat()
    ).subscribe();
  }

  async ngOnDestroy() {
    this.destroy$.next(true);
  }

  async cancelTask(event: Event, id) {
    event.stopImmediatePropagation();
    event.preventDefault();

    const alert = await this.alertController.create({
      buttons: [
        'Back',
        {
          text: 'Cancel Job',
          handler: async () => {
            const loader = await this.loadingController.create({message: 'Canceling task...'});
            await loader.present();

            try {
              await (await this.api.cancelTask(id)).toPromise();
            } catch (error) {
              (await this.alertController.create({message: 'Unable to cancel the task', buttons: ['Dismiss']})).present();
            }
            await loader.dismiss();
          }
        }
      ],
      message: 'Are you sure that you want to cancel this job?'
    });

    await alert.present();
  }

  async showDetails(event: Event, parameters) {
    event.stopImmediatePropagation();
    event.preventDefault();

    const modal = await this.modalController.create({
      cssClass: 'wide-modal',
      component: TrainComponent,
      componentProps: {parameters}
    });

    await modal.present();
  }
}
