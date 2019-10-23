import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-use-model',
  templateUrl: './use-model.component.html',
  styleUrls: ['./use-model.component.scss'],
})
export class UseModelComponent implements OnInit {
  @Input() features: string[];
  testForm: FormGroup;
  result;

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder,
  ) {}

  ngOnInit() {
    this.testForm = this.formBuilder.group({
      inputs: this.formBuilder.array(
        new Array(this.features.length).fill(['', Validators.required])
      )
    });
  }

  testModel() {
    const data = new FormData();
    data.append('data', this.testForm.get('inputs').value);
    data.append('features', JSON.stringify(this.features));

    this.backend.testModel(data).subscribe(
      (result) => {
        this.result = result;
      },
      () => {
        this.result = undefined;
      }
    );
  }

  exportModel() {
    window.open('http://127.0.0.1:5000/export-model', '_self');
  }

  exportPMML() {
    window.open('http://127.0.0.1:5000/export-pmml', '_self');
  }
}
