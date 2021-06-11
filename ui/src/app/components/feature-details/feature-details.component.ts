import { Component, Input, OnInit } from '@angular/core';

import { DataAnalysis } from '../../interfaces';

@Component({
  selector: 'app-feature-details',
  templateUrl: './feature-details.component.html',
  styleUrls: ['./feature-details.component.scss'],
})
export class FeatureDetailsComponent implements OnInit {
  @Input() data: DataAnalysis;
  @Input() label: string;
  @Input() type: string;

  features: string[];

  ngOnInit() {
    this.features = Object.keys(this.data.summary).filter(item => item !== this.label);
  }
}
