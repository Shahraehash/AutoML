import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { DatePipe } from '@angular/common';
import { MatTableDataSource } from '@angular/material';
import { LoadingController, AlertController } from '@ionic/angular';

import { BackendService } from '../../services/backend.service';
import { DataAnalysisReply, Jobs } from '../../interfaces';

@Component({
  selector: 'app-explore',
  templateUrl: './explore.component.html',
  styleUrls: ['./explore.component.scss'],
})
export class ExploreComponent implements OnInit {
  @Output() stepFinished = new EventEmitter();
  @Output() reset = new EventEmitter();

  analysis: DataAnalysisReply;
  jobs: MatTableDataSource<Jobs>;
  columns = ['Date', 'Status', 'Actions'];
  constructor(
    public backend: BackendService,
    private alertController: AlertController,
    private datePipe: DatePipe,
    private loadingController: LoadingController
  ) {}

  ngOnInit() {
    if (!this.backend.currentDatasetId) {
      return;
    }

    this.backend.getDataAnalysis().subscribe(data => this.analysis = data);
    this.updateJobs();
  }

  getValue(job, column) {
    switch (column) {
      case 'Date':
        return this.datePipe.transform(job.date, 'shortDate');
      case 'Status':
        return job.metadata.date ? 'Completed' : 'Pending';
    }
  }

  useJob(id, step) {
    this.backend.currentJobId = id;
    this.stepFinished.emit({nextStep: step, data: Object.keys(this.analysis.analysis.train.summary).length});
  }

  async deleteJob(id) {
    const alert = await this.alertController.create({
      buttons: [
        'Dismiss',
        {
          text: 'Delete',
          handler: async () => {
            const loading = await this.loadingController.create({
              message: 'Deleting job...'
            });
            await loading.present();
            await this.backend.deleteJob(id).toPromise();
            this.updateJobs();
            await loading.dismiss();
          }
        }
      ],
      header: 'Are you sure you want to delete this?',
      subHeader: 'This cannot be undone.',
      message: 'Are you sure you want to delete the selected run?'
    });
    await alert.present();
  }

  async deleteDataset() {
    const alert = await this.alertController.create({
      buttons: [
        'Dismiss',
        {
          text: 'Delete',
          handler: async () => {
            const loading = await this.loadingController.create({
              message: 'Deleting dataset...'
            });
            await loading.present();
            await this.backend.deleteDataset(this.backend.currentDatasetId).toPromise();
            this.reset.emit();
            await loading.dismiss();
          }
        }
      ],
      header: 'Are you sure you want to delete this?',
      subHeader: 'This cannot be undone.',
      message: 'Are you sure you want to delete this dataset?'
    });
    await alert.present();
  }

  async newJob() {
    const loading = await this.loadingController.create({message: 'Creating new job...'});
    await loading.present();
    await this.backend.createJob();
    this.stepFinished.emit({nextStep: 'train', data: Object.keys(this.analysis.analysis.train.summary).length});
    await loading.dismiss();
  }

  private async updateJobs() {
    this.jobs = new MatTableDataSource(
      (await this.backend.getJobs().toPromise()).filter(job => job.metadata.datasetid === this.backend.currentDatasetId)
    );
  }
}
