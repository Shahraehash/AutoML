import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';

@Component({
  selector: 'app-textarea-modal',
  templateUrl: './textarea-modal.component.html',
  styleUrls: ['./textarea-modal.component.scss'],
})
export class TextareaModalComponent implements OnInit {
  @Input() header: string;
  @Input() subHeader: string;
  @Input() message: string;
  @Input() buttons: {name: string, handler?: () => void}[];
  @Input() inputs: {name: string; placeholder: string}[];

  parsedInputs: FormGroup;

  constructor(
    private formBuilder: FormBuilder
  ) {}

  ngOnInit() {
    this.parsedInputs = this.formBuilder.group(
      this.inputs.reduce(
        (obj, item) => {
          obj[item.name] = '';
          return obj;
        },
        {}
      )
    );
  }
}
