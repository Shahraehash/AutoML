import { Component, Input, OnInit } from '@angular/core';
import { AlertController, ModalController, LoadingController } from '@ionic/angular';
import { ReplaySubject } from 'rxjs';
import { delay, repeat, tap, takeUntil } from 'rxjs/operators';

import { MiloApiService } from '../../services/milo-api/milo-api.service';
import { PendingTasks } from '../../interfaces';
import { TrainComponent } from '../train/train.component';

@Component({
  selector: 'app-pending-tasks',
  templateUrl: './pending-tasks.component.html',
  styleUrls: ['./pending-tasks.component.scss'],
})
export class PendingTasksComponent implements OnInit {
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
      takeUntil(this.destroy$),
      tap(pending => this.pendingTasks = pending),
      delay(5000),
      repeat()
    ).subscribe();
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
