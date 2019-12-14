import { Component, Input, HostBinding, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
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
    private formBuilder: FormBuilder,
    private sanitizer: DomSanitizer
  ) {}

  @HostBinding('attr.style')
  get textAreaHeight() {
    return this.sanitizer.bypassSecurityTrustStyle(`--textarea-height: calc(${100 / this.inputs.length}% - ${25 * this.inputs.length}px)`);
  }

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

  buttonHandler(handler) {
    if (!handler) {
      return;
    }

    handler(this.parsedInputs.value);
  }
}
