import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-textarea-modal',
  templateUrl: './textarea-modal.component.html',
  styleUrls: ['./textarea-modal.component.scss'],
})
export class TextareaModalComponent implements OnInit {
  @Input() header: string;
  @Input() subHeader: string;
  @Input() message: string;
  @Input() buttons: {name: string}[];
  @Input() inputs: {name: string; placeholder: string}[];

  constructor() {}

  ngOnInit() {
  }
}
