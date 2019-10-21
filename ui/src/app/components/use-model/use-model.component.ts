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

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder,
  ) {}

  ngOnInit() {
    this.testForm = this.formBuilder.group({
      inputs: this.formBuilder.array(this.features, Validators.required)
    });
  }

  testModel() {
    const data = new FormData();
    data.append('data', this.testForm.get('inputs').value);
    data.append('features', JSON.stringify(this.features));

    this.backend.testModel(data).subscribe();
  }
}
