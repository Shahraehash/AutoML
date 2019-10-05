import { Component, ViewChild, OnInit } from '@angular/core';
import { MatSort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';
import { AlertController } from '@ionic/angular';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss'],
})
export class ResultsPage implements OnInit {
  data;
  results;
  columns: {key: string; name: string; number?: boolean, hideMobile?: boolean}[] = [
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
      name: 'Scaler',
      hideMobile: true
    },
    {
      key: 'feature_selector',
      name: 'Feature Selector',
      hideMobile: true
    },
    {
      key: 'scorer',
      name: 'Scorer',
      hideMobile: true
    },
    {
      key: 'searcher',
      name: 'Searcher',
      hideMobile: true
    }
  ];

  @ViewChild(MatSort, {static: false}) sort: MatSort;

  constructor(
    private alertController: AlertController,
    private backend: BackendService,
  ) {}

  ngOnInit() {
    this.backend.getResults().subscribe(
      data => {
        this.data = data;
        this.results = new MatTableDataSource(data);
        setTimeout(() => {
          this.results.sort = this.sort;
        }, 1);
     },
      async () => {
        const alert = await this.alertController.create({
          header: 'Unable to Load Results',
          message: 'Please make sure the backend is reachable and try again.',
          buttons: ['Dismiss']
        });

        await alert.present();
      }
    );
  }

  getColumns() {
    const isMobile = window.screen.width < 420;
    return isMobile ? this.columns.filter(c => !c.hideMobile).map(c => c.key) : this.columns.map(c => c.key);
  }

  export() {
    window.open('http://127.0.0.1:5000/export', '_self');
  }
}
