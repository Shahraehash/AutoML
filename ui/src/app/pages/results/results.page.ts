import { Component, ViewChild, OnInit } from '@angular/core';
import { MatSort } from '@angular/material/sort';
import { MatTableDataSource } from '@angular/material/table';
import { AlertController } from '@ionic/angular';

import { BackendService } from '../../services/backend.service';
import { GeneralizationResult } from '../../interfaces';

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss'],
})
export class ResultsPage implements OnInit {
  activeRow = 0;
  data;
  rocData;
  sortedData;
  trainingRocData;
  results: MatTableDataSource<GeneralizationResult>;
  columns: {key: string; name: string; number?: boolean, hideMobile?: boolean}[] = [
    {
      key: 'estimator',
      name: 'Estimator'
    },
    {
      key: 'auc',
      name: 'AUC',
      number: true
    },
    {
      key: 'accuracy',
      name: 'Accuracy',
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
      key: 'brier_score',
      name: 'Brier Score',
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
    },
    {
      key: 'actions',
      name: ''
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

        this.results.connect().subscribe(d => {
          this.sortedData = d;
        });
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
    const isMobile = window.innerWidth < 1350;
    return isMobile ? this.columns.filter(c => !c.hideMobile).map(c => c.key) : this.columns.map(c => c.key);
  }

  export() {
    window.open('http://127.0.0.1:5000/export', '_self');
  }

  parse(object, mode) {
    let fpr;
    let tpr;
    let upper;
    let lower;
    const textElements = [
      'Estimator: ' + object.estimator,
      'Scaler: ' + object.scaler,
      'Selector: ' + object.feature_selector,
      'Scorer: ' + object.scorer,
      'Searcher: ' + object.searcher
    ];

    if (mode === 'generalization') {
      fpr = JSON.parse(object.generalization_fpr);
      tpr = JSON.parse(object.generalization_tpr);
    } else if (mode === 'reliability') {
      fpr = JSON.parse(object.mpv);
      tpr = JSON.parse(object.fop);
    } else if (mode === 'mean') {
      fpr = JSON.parse(object.mean_fpr);
      tpr = JSON.parse(object.mean_tpr);
      upper = JSON.parse(object.mean_upper);
      lower = JSON.parse(object.mean_lower);
    } else if (mode === 'test') {
      fpr = JSON.parse(object.test_fpr);
      tpr = JSON.parse(object.test_tpr);
    } else {
      return;
    }

    if (mode === 'reliability') {
      textElements.push('Brier Score: ' + object.brier_score.toFixed(4));
    } else {
      textElements.push('AUC = ' + this.calculateArea(tpr, fpr) + (mode === 'mean' ? ' ± ' + object.std_auc.toFixed(4) : ''));
    }

    return {
      fpr,
      tpr,
      upper,
      lower,
      textElements
    };
  }

  launchModel(index: number) {
    console.log(index);
  }

  private calculateArea(tpr, fpr) {
    let area = 0.0;
    tpr.forEach((_, i) => {
      if ('undefined' !== typeof fpr[i - 1]) {
        area += (fpr[i] - fpr[i - 1]) * (tpr[i - 1] + tpr[i]) / 2;
      }
    });
    return area.toFixed(4);
  }
}
