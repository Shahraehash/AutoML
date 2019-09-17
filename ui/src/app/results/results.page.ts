import { Component, ViewChild, OnInit } from '@angular/core';
import { MatSort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';

import { BackendService } from '../services/backend.service';

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss'],
})
export class ResultsPage implements OnInit {
  results;
  displayedColumns: string[] = ['pipeline', 'accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 'best_params'];

  @ViewChild(MatSort, {static: false}) sort: MatSort;

  constructor(
    private backend: BackendService,
  ) {}

  ngOnInit() {
    this.backend.getResults().subscribe(data => {
      this.results = new MatTableDataSource(data);
      setTimeout(() => {
        this.results.sort = this.sort;
      }, 1);
    });
  }
}
