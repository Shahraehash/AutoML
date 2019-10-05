import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

import { parse } from 'papaparse';
import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit {
  labels = [];
  uploadForm: FormGroup;
  estimators = [
    {
      label: 'Gradient Boosting Machine',
      value: 'gb'
    },
    {
      label: 'K-nearest Neighbor',
      value: 'knn'
    },
    {
      label: 'Logistic Regression',
      value: 'lr'
    },
    {
      label: 'Neural Network',
      value: 'mlp'
    },
    {
      label: 'naive Bayes',
      value: 'nb'
    },
    {
      label: 'Random Forest',
      value: 'rf'
    },
    {
      label: 'Support Vector Machine',
      value: 'svm'
    }
  ];
  scalers = [
    {
      label: 'None',
      value: 'none'
    },
    {
      label: 'Standard',
      value: 'std'
    },
    {
      label: 'Min Max',
      value: 'minmax'
    }
  ];
  featureSelectors = [
    {
      label: 'None',
      value: 'none'
    },
    {
      label: 'Principal Component Analysis (80%)',
      value: 'pca-80'
    },
    {
      label: 'Principal Component Analysis (90%)',
      value: 'pca-90'
    },
    {
      label: 'Random Forest Importance (25%)',
      value: 'rf-25'
    },
    {
      label: 'Random Forest Importance (50%)',
      value: 'rf-50'
    },
    {
      label: 'Random Forest Importance (75%)',
      value: 'rf-75'
    },
    {
      label: 'Select Percentile (25%)',
      value: 'select-25'
    },
    {
      label: 'Select Percentile (50%)',
      value: 'select-50'
    },
    {
      label: 'Select Percentile (75%)',
      value: 'select-75'
    }
  ];
  searchers = [
    {
      label: 'Grid',
      value: 'grid'
    },
    {
      label: 'Random',
      value: 'random'
    }
  ];
  scorers = [
    {
      label: 'Accuracy',
      value: 'accuracy'
    },
    {
      label: 'ROC AUC',
      value: 'roc_auc'
    },
    {
      label: 'F1',
      value: 'f1_macro'
    }
  ];

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder,
    private router: Router
  ) {}

  ngOnInit() {
    this.uploadForm = this.formBuilder.group({
      label_column: ['', Validators.required],
      train: ['', Validators.required],
      test: ['', Validators.required],
      estimators: this.formBuilder.array(this.estimators),
      scalers: this.formBuilder.array(this.scalers),
      featureSelectors: this.formBuilder.array(this.featureSelectors),
      searchers: this.formBuilder.array(this.searchers),
      scorers: this.formBuilder.array(this.scorers)
    });
  }

  onSubmit() {
    const formData = new FormData();
    formData.append('train', this.uploadForm.get('train').value);
    formData.append('test', this.uploadForm.get('test').value);
    formData.append('label_column', this.uploadForm.get('label_column').value);
    formData.append('ignore_estimators', this.getValues('estimators'));
    formData.append('ignore_scalers', this.getValues('scalers'));
    formData.append('ignore_feature_selectors', this.getValues('featureSelectors'));
    formData.append('ignore_searchers', this.getValues('searchers'));
    formData.append('ignore_scorers', this.getValues('scorers'));

    this.backend.submitData(formData).subscribe(
      () => this.router.navigate(['/train', {upload: true}]),
      (err) => console.log(err)
    );

    return false;
  }

  onFileSelect(event) {
    if (event.target.files.length === 1) {
      const file = event.target.files[0];

      if (!this.labels.length) {
        parse(file, {
          complete: reply => this.labels = reply.data[0]
        });
      }

      this.uploadForm.get(event.target.name).setValue(file);
    }
  }

  private getValues(key) {
    return this.uploadForm.get(key).value.flatMap((value, index) => {
      return value ? [] : this.estimators[index].value;
    });
  }
}
