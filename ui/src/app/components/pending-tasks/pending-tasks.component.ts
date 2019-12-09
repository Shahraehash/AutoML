import { Component, Input, OnInit } from '@angular/core';
import { AlertController, LoadingController } from '@ionic/angular';
import { Observable, timer } from 'rxjs';
import { switchMap } from 'rxjs/operators';

import { PendingTasks } from '../../interfaces';
import { BackendService } from 'src/app/services/backend.service';

@Component({
  selector: 'app-pending-tasks',
  templateUrl: './pending-tasks.component.html',
  styleUrls: ['./pending-tasks.component.scss'],
})
export class PendingTasksComponent implements OnInit {
  @Input() firstViewData: PendingTasks;
  pendingTasks$: Observable<PendingTasks>;

  constructor(
    private alertController: AlertController,
    private backend: BackendService,
    private loadingController: LoadingController
  ) {}

  ngOnInit() {
    this.pendingTasks$ = timer(0, 150000000).pipe(
      switchMap(() => this.backend.getPendingTasks())
    );
  }

  async cancelTask(id) {
    const alert = await this.alertController.create({
      buttons: [
        'Back',
        {
          text: 'Cancel Job',
          handler: async () => {
            const loader = await this.loadingController.create({message: 'Canceling Task...'});
            await loader.present();

            try {
              await this.backend.cancelTask(id).toPromise();
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
}
