import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';

import * as pipelineOptions from './pipeline.processors.json';
import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-train',
  templateUrl: 'train.page.html',
  styleUrls: ['train.page.scss']
})
export class TrainPage implements OnInit {
  uploadComplete = false;
  training = false;
  results;
  trainForm: FormGroup;
  pipelineProcessors = (pipelineOptions as any).default;

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder,
    private route: ActivatedRoute
  ) {}

  ngOnInit() {
    this.uploadComplete = this.route.snapshot.params.upload;
    this.trainForm = this.formBuilder.group({
      estimators: this.formBuilder.array(this.pipelineProcessors.estimators),
      scalers: this.formBuilder.array(this.pipelineProcessors.scalers),
      featureSelectors: this.formBuilder.array(this.pipelineProcessors.featureSelectors),
      searchers: this.formBuilder.array(this.pipelineProcessors.searchers),
      scorers: this.formBuilder.array(this.pipelineProcessors.scorers)
    });
  }

  startTraining() {
    this.training = true;

    const formData = new FormData();
    formData.append('ignore_estimator', this.getValues('estimators').join(','));
    formData.append('ignore_scaler', this.getValues('scalers').join(','));
    formData.append('ignore_feature_selector', this.getValues('featureSelectors').join(','));
    formData.append('ignore_searcher', this.getValues('searchers').join(','));
    formData.append('ignore_scorer', this.getValues('scorers').join(','));

    this.backend.startTraining(formData).subscribe(
      (res) => {
        this.training = false;
        this.results = res;
      }
    );
  }

  private getValues(key) {
    return this.trainForm.get(key).value.flatMap((value, index) => {
      return value ? [] : this.pipelineProcessors[key][index].value;
    });
  }
}
