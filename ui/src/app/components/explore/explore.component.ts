import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { DatePipe } from '@angular/common';
import { MatTableDataSource } from '@angular/material';

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
  columns = ['Date', 'Completed', 'actions'];
  constructor(
    public backend: BackendService,
    private datePipe: DatePipe
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
      case 'Completed':
        return this.datePipe.transform(job.metadata.date, 'shortTime');
    }
  }

  continue() {
    this.backend.createJob().then(_ => {
      this.stepFinished.emit({nextStep: 'train'});
    });
  }
}
