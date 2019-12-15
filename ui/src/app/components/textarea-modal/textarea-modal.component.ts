import { Component, Input, HostBinding, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { FormBuilder, FormGroup } from '@angular/forms';
import { ModalController } from '@ionic/angular';

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
  @Input() inputs: {
    name: string;
    placeholder: string;
    disabled?: boolean;
    value?: string
  }[];

  parsedInputs: FormGroup;

  constructor(
    private formBuilder: FormBuilder,
    private modalController: ModalController,
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
          obj[item.name] = [{value: item.value, disabled: item.disabled}];
          return obj;
        },
        {}
      )
    );
  }

  async buttonHandler(handler) {
    if (!handler) {
      await this.modalController.dismiss();
      return;
    }

    if (handler(this.parsedInputs.value) !== false) {
      await this.modalController.dismiss();
    }
  }
}
