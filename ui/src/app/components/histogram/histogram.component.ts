import { Component, ElementRef, Input, OnInit } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-histogram',
  templateUrl: './histogram.component.html',
  styleUrls: ['./histogram.component.scss'],
})
export class HistogramComponent implements OnInit {
  @Input() data;
  private svg;

  constructor(
    private element: ElementRef
  ) { }

  ngOnInit() {
    this.svg = d3.select(this.element.nativeElement).select('svg');

    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = 960 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const x = d3.scaleBand()
      .range([0, width])
      .padding(0.1);
    const y = d3.scaleLinear()
      .range([height, 0]);

    this.svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform',
        'translate(' + margin.left + ',' + margin.top + ')');

    x.domain(this.data.map(d => d[0]));
    y.domain([0, d3.max(this.data, d => d[1])]);

    // append the rectangles for the bar chart
    svg.selectAll(".bar")
      .data(data)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", function (d) { return x(d.salesperson); })
      .attr("width", x.bandwidth())
      .attr("y", function (d) { return y(d.sales); })
      .attr("height", function (d) { return height - y(d.sales); });

    // add the x Axis
    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x));

    // add the y Axis
    svg.append("g")
      .call(d3.axisLeft(y));






  }
}
