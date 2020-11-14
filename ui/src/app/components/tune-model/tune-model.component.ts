import { Component, Input } from '@angular/core';
import { PopoverController } from '@ionic/angular';

@Component({
  selector: 'app-tune-model',
  templateUrl: './tune-model.component.html',
  styleUrls: ['./tune-model.component.scss'],
})
export class TuneModelComponent {
  @Input() threshold = .5;

  constructor(
    public popoverController: PopoverController
  ) {}

  submit() {
    this.popoverController.dismiss({threshold: this.threshold});
  }
}
