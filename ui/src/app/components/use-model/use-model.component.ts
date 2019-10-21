import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-use-model',
  templateUrl: './use-model.component.html',
  styleUrls: ['./use-model.component.scss'],
})
export class UseModelComponent implements OnInit {
  @Input() features: string[];
  testForm: FormGroup;

  constructor(
    private formBuilder: FormBuilder,
  ) {}

  ngOnInit() {
    this.testForm = this.formBuilder.group({
      inputs: this.formBuilder.array(this.features, Validators.required)
    });
  }
}
