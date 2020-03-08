import { Component, Input } from '@angular/core';

import { RefitGeneralization } from 'src/app/interfaces';

@Component({
  selector: 'app-model-statistics',
  templateUrl: './model-statistics.component.html',
  styleUrls: ['./model-statistics.component.scss'],
})
export class ModelStatisticsComponent {
    @Input() generalization: RefitGeneralization;
}
