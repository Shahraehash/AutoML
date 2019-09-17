import { Component, OnInit } from '@angular/core';

import { BackendService } from '../services/backend.service';

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss'],
})
export class ResultsPage implements OnInit {
  results;

  constructor(
    private backend: BackendService,
  ) {}

  ngOnInit() {
    this.backend.getResults().subscribe(data => this.results = data);
  }

}
