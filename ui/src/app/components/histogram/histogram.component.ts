import { Component, ElementRef, Input, OnInit } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-histogram',
  styleUrls: ['./histogram.component.scss'],
  template: '<svg class="histogram" viewBox="0 0 300 240"></svg>'
})
export class HistogramComponent implements OnInit {
  @Input() positiveData: [number[], number[]];
  @Input() negativeData: [number[], number[]];
  private svg;

  constructor(
    private element: ElementRef
  ) {}

  ngOnInit() {
    this.svg = d3.select(this.element.nativeElement).select('svg');
    const margin = { top: 0, right: 0, bottom: 0, left: 0 };
    const width = 300 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;
    this.svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    let point;
    const positiveData = [];
    const negativeData = [];

    this.positiveData[0].push(0);
    this.negativeData[0].push(0);

    for (let i = 0; i < this.positiveData[1].length; i++) {
      point = [this.positiveData[1][i], this.positiveData[0][i]];
      positiveData.push(point);
    }

    for (let i = 0; i < this.negativeData[1].length; i++) {
      point = [this.negativeData[1][i], this.negativeData[0][i]];
      negativeData.push(point);
    }

    const allData = positiveData.concat(negativeData);

    const x = d3.scaleLinear()
      .domain([d3.min(allData, d => d[0]), d3.max(allData, d => d[0])])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(allData, d => d[1])])
      .range([height, 0]);

    this.svg.append('g')
      .attr('transform', 'translate(0,' + height + ')')
      .call(d3.axisBottom(x));

    const positiveBar = this.svg.selectAll('.bar.positive')
      .data(positiveData)
      .enter().append('g')
      .attr('class', 'bar positive')
      .attr('transform', d => 'translate(' + x(d[0]) + ',' + y(d[1]) + ')');

    positiveBar.append('rect')
      .attr('x', 1)
      .attr('width', Math.abs(width / positiveData.length - 1))
      .attr('height', d => height - y(d[1]));

    const negativeBar = this.svg.selectAll('.bar.negative')
      .data(negativeData)
      .enter().append('g')
      .attr('class', 'bar negative')
      .attr('transform', d => 'translate(' + x(d[0]) + ',' + y(d[1]) + ')');

      negativeBar.append('rect')
      .attr('x', 1)
      .attr('width', Math.abs(width / negativeData.length - 1))
      .attr('height', d => height - y(d[1]));
  }
}
