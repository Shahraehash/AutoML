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
  columns: {key: string; name: string; number?: boolean}[] = [
    {
      key: 'estimator',
      name: 'Estimator'
    },
    {
      key: 'accuracy',
      name: 'Accuracy',
      number: true
    },
    {
      key: 'auc',
      name: 'AUC',
      number: true
    },
    {
      key: 'f1',
      name: 'F1',
      number: true
    },
    {
      key: 'sensitivity',
      name: 'Sensitivity',
      number: true
    },
    {
      key: 'specificity',
      name: 'Specificity',
      number: true
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

  export() {
    window.open('http://127.0.0.1:5000/export', '_self');
  }
}
