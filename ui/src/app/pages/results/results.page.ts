import { Component, ViewChild, OnInit } from '@angular/core';
import { MatSort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss'],
})
export class ResultsPage implements OnInit {
  data;
  results;
  columns: {key: string; name: string}[] = [
    {
      key: 'estimator',
      name: 'Estimator'
    },
    {
      key: 'accuracy',
      name: 'Accuracy'
    },
    {
      key: 'auc',
      name: 'AUC'
    },
    {
      key: 'f1',
      name: 'F1'
    },
    {
      key: 'sensitivity',
      name: 'Sensitivity'
    },
    {
      key: 'specificity',
      name: 'Specificity'
    },
    {
      key: 'scaler',
      name: 'Scaler'
    },
    {
      key: 'feature_selector',
      name: 'Feature Selector'
    },
    {
      key: 'scorer',
      name: 'Scorer'
    },
    {
      key: 'searcher',
      name: 'Searcher'
    }
  ];
  displayedColumns = this.columns.map(c => c.key);

  @ViewChild(MatSort, {static: false}) sort: MatSort;

  constructor(
    private backend: BackendService,
  ) {}

  ngOnInit() {
    this.backend.getResults().subscribe(data => {
      this.data = data;
      this.results = new MatTableDataSource(data);
      setTimeout(() => {
        this.results.sort = this.sort;
      }, 1);
    });
  }
}
