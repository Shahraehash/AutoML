import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { DatePipe } from '@angular/common';
import { MatTableDataSource } from '@angular/material';
import { LoadingController } from '@ionic/angular';

import { BackendService } from '../../services/backend.service';
import { DataAnalysisReply, Jobs } from '../../interfaces';

@Component({
  selector: 'app-explore',
  templateUrl: './explore.component.html',
  styleUrls: ['./explore.component.scss'],
})
export class ExploreComponent implements OnInit {
  @Output() stepFinished = new EventEmitter();

  analysis: DataAnalysisReply;
  jobs: MatTableDataSource<Jobs>;
  columns = ['Date', 'Status', 'actions'];
  constructor(
    public backend: BackendService,
    private datePipe: DatePipe,
    private loadingController: LoadingController
  ) {}

  ngOnInit() {
    if (!this.backend.currentDatasetId) {
      return;
    }

    this.backend.getDataAnalysis().subscribe(data => this.analysis = data);
    this.backend.getJobs().subscribe(data => {
      this.jobs = new MatTableDataSource(
        data.filter(job => job.metadata.datasetid === this.backend.currentDatasetId)
      );
    });
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
    this.stepFinished.emit({nextStep: step});
  }

  deleteJob(id) {
    console.log(id);
  }

  deleteDataset() {
    console.log('delete');
  }

  async newJob() {
    const loading = await this.loadingController.create({message: 'Creating new job...'});
    await loading.present();
    await this.backend.createJob();
    this.stepFinished.emit({nextStep: 'train'});
    await loading.dismiss();
  }
}
