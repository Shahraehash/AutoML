import { Component, Input } from '@angular/core';
import { DataAnalysisSummary } from 'src/app/interfaces';

@Component({
  selector: 'app-feature-analysis',
  templateUrl: './feature-analysis.component.html',
  styleUrls: ['./feature-analysis.component.scss'],
})
export class FeatureAnalysisComponent {
  @Input() summary: DataAnalysisSummary;
  @Input() mode: number;
  @Input() median: number;
  @Input() nullCount: number;
  @Input() isValid: boolean;

  constructor() {}
}
