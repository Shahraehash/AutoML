import { Component, OnInit } from '@angular/core';

import { BackendService } from '../../services/backend.service';
import { DataAnalysisReply } from '../../interfaces';

@Component({
  selector: 'app-explore',
  templateUrl: './explore.component.html',
  styleUrls: ['./explore.component.scss'],
})
export class ExploreComponent implements OnInit {
  data: DataAnalysisReply;
  constructor(
    public backend: BackendService
  ) {}

  ngOnInit() {
    this.backend.getDataAnalysis().subscribe(data => this.data = data);
  }
}
