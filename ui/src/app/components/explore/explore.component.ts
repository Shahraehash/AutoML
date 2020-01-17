import { Component, EventEmitter, OnInit, Output } from '@angular/core';

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
  jobs: Jobs[];
  columns = ['Date', 'Completed', 'actions'];
  constructor(
    public backend: BackendService
  ) {}

  ngOnInit() {
    if (!this.backend.currentDatasetId) {
      return;
    }

    this.backend.getDataAnalysis().subscribe(data => this.analysis = data);
    this.backend.getJobs().subscribe(data => this.jobs = data);
  }

  continue() {
    this.backend.createJob().then(_ => {
      this.stepFinished.emit({nextStep: 'train'});
    });
  }
}
