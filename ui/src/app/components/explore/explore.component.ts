import { Component, EventEmitter, OnInit, Output } from '@angular/core';

import { BackendService } from '../../services/backend.service';
import { DataAnalysisReply } from '../../interfaces';

@Component({
  selector: 'app-explore',
  templateUrl: './explore.component.html',
  styleUrls: ['./explore.component.scss'],
})
export class ExploreComponent implements OnInit {
  @Output() stepFinished = new EventEmitter();

  data: DataAnalysisReply;
  constructor(
    public backend: BackendService
  ) {}

  ngOnInit() {
    this.backend.getDataAnalysis().subscribe(data => this.data = data);
  }

  continue() {
    this.backend.createJob().then(_ => {
      this.stepFinished.emit({state: 'explore'});
    });
  }
}
