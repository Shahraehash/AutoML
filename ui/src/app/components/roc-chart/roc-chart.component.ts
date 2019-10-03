import { Component, Input, OnInit } from '@angular/core';

import { ROCChartService } from '../../services/roc-chart.service';
import * as sampleData from './sample-data.json';

@Component({
  selector: 'app-roc-chart',
  templateUrl: './roc-chart.component.html',
  styleUrls: ['./roc-chart.component.scss'],
})
export class RocChartComponent implements OnInit {
  @Input() data;

  private margin = {top: 30, right: 61, bottom: 70, left: 61};
  private width = 470 - this.margin.left - this.margin.right;
  private height = 450 - this.margin.top - this.margin.bottom;
  private rocChartOptions = {
    margin: this.margin,
    width: this.width,
    height: this.height
  };

  constructor(
    private roc: ROCChartService
  ) {}

  ngOnInit() {
    this.roc.create(this.data, this.rocChartOptions);
  }
}
