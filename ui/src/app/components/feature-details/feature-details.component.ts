import { Component, Input } from '@angular/core';

import { DataAnalysis } from '../../interfaces';

@Component({
  selector: 'app-feature-details',
  templateUrl: './feature-details.component.html',
  styleUrls: ['./feature-details.component.scss'],
})
export class FeatureDetailsComponent {
  @Input() data: DataAnalysis;
  @Input() label: string;
  @Input() type: string;

  constructor() {}

  preventSort() {
    return 0;
  }
}
