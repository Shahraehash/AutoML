import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-tune-model',
  templateUrl: './tune-model.component.html',
  styleUrls: ['./tune-model.component.scss'],
})
export class TuneModelComponent {
  @Input() threshold = .5;
}
