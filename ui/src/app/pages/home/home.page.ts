import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { parse } from 'papaparse';

import * as pipelineOptions from './pipeline.processors.json';
import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit {
  labels = [];
  uploadForm: FormGroup;
  pipelineProcessors = (pipelineOptions as any).default;

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
      estimators: this.formBuilder.array(this.pipelineProcessors.estimators),
      scalers: this.formBuilder.array(this.pipelineProcessors.scalers),
      featureSelectors: this.formBuilder.array(this.pipelineProcessors.featureSelectors),
      searchers: this.formBuilder.array(this.pipelineProcessors.searchers),
      scorers: this.formBuilder.array(this.pipelineProcessors.scorers)
    });
  }

  onSubmit() {
    const formData = new FormData();
    formData.append('train', this.uploadForm.get('train').value);
    formData.append('test', this.uploadForm.get('test').value);
    formData.append('label_column', this.uploadForm.get('label_column').value);
    formData.append('ignore_estimator', this.getValues('estimators').join(','));
    formData.append('ignore_scaler', this.getValues('scalers').join(','));
    formData.append('ignore_feature_selector', this.getValues('featureSelectors').join(','));
    formData.append('ignore_searcher', this.getValues('searchers').join(','));
    formData.append('ignore_scorer', this.getValues('scorers').join(','));

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
      return value ? [] : this.pipelineProcessors[key][index].value;
    });
  }
}
