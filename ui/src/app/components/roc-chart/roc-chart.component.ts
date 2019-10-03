import { Component, Input, OnInit } from '@angular/core';
import * as d3 from 'd3';
import { schemeCategory10 } from 'd3-scale-chromatic';
import { scaleLinear, scaleOrdinal } from 'd3-scale';
import * as d3Axis from 'd3-axis';

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

  constructor() {}

  ngOnInit() {
    this.create(this.data, this.rocChartOptions);
  }

  private create(data, options) {
      const cfg = {...{
          margin: {top: 30, right: 20, bottom: 70, left: 61},
          width: 470,
          height: 450,
          tickValues: [0, .1, .25, .5, .75, .9, 1]
      }, ...options};

      const format = d3.format('.2');
      const aucFormat = d3.format('.4r');

      const x = scaleLinear().range([0, cfg.width]);
      const y = scaleLinear().range([cfg.height, 0]);
      const color = scaleOrdinal(schemeCategory10);

      const xAxis = d3Axis.axisBottom(x);
      const yAxis = d3Axis.axisLeft(y);

      xAxis.tickValues(cfg.tickValues);
      yAxis.tickValues(cfg.tickValues);

      // Apply the format to the ticks we chose
      xAxis.tickFormat(format);
      yAxis.tickFormat(format);

      // A function that returns a line generator
      const curve = (points) => {

          const lineGenerator = d3.line()
              .curve(d3.curveBasis)
              .x((d) => x(d[0]))
              .y((d) => y(d[1]));

          return lineGenerator(points);
      };

      // A function that returns an area generator
      const areaUnderCurve = (points) => {

          const areaGenerator = d3.area()
              .x((d) => x(d[0]))
              .y0(cfg.height)
              .y1((d) => y(d[1]));

          return areaGenerator(points);
      };

      const svg = d3.select('#roc')
          .append('svg')
          .attr('width', cfg.width + cfg.margin.left + cfg.margin.right)
          .attr('height', cfg.height + cfg.margin.top + cfg.margin.bottom)
          .append('g')
              .attr('transform', 'translate(' + cfg.margin.left + ',' + cfg.margin.top + ')');

      x.domain([0, 1]);
      y.domain([0, 1]);

      svg.append('g')
          .attr('class', 'x axis')
          .attr('transform', 'translate(0,' + cfg.height + ')')
          .call(xAxis)
          .append('text')
              .attr('x', cfg.width / 2)
              .attr('y', 40 )
              .style('text-anchor', 'middle')
              .text('False Positive Rate');


      svg.append('g')
          .attr('class', 'y axis')
          .call(yAxis)
          .append('text')
          .attr('transform', 'rotate(-90)')
          .attr('y', -35)
          .attr('x', 0 - cfg.height / 2.8)
          .style('font-size', '12px')
          .style('text-anchor', 'left')
          .text('True Positive Rate');

      // draw the random guess line
      svg.append('line')
          .attr('class', 'curve')
          .style('stroke', 'black')
          .attr('x1', 0)
          .attr('x2', cfg.width)
          .attr('y1', cfg.height)
          .attr('y2', 0)
          .style('stroke-width', 2)
          .style('stroke-dasharray', 8)
          .style('opacity', .4);

      // Draw the ROC curves
      const drawCurve = (key, points, stroke) => {
          svg.append('path')
              .attr('class', 'curve')
              .style('stroke', stroke)
              .attr('d', curve(points))
              .on('mouseover', () => {
                  const areaID = '#' + key + '-area';
                  svg.select(areaID).style('opacity', .4);

                  const aucText = '.' + key + '-text';
                  svg.selectAll(aucText).style('opacity', .9);
              })
              .on('mouseout', () => {
                  const areaID = '#' + key + '-area';
                  svg.select(areaID).style('opacity', 0);

                  const aucText = '.' + key + '-text';
                  svg.selectAll(aucText).style('opacity', 0);
              });
      };

      // Draw the area under the ROC curves
      const drawArea = (key, points, fill) => {
          svg.append('path')
          .attr('d', areaUnderCurve(points))
          .attr('class', 'area')
          .attr('id', key + '-area')
          .style('fill', fill)
          .style('opacity', 0);
      };

      const drawAUCText = (key, auc) => {
          svg.append('g')
          .attr('class', key + '-text')
          .style('opacity', 0)
          .attr('transform', 'translate(' + .5 * cfg.width + ',' + .79 * cfg.height + ')')
          .append('text')
              .text(key.replace(/__/g, ' '))
              .style('fill', 'white')
              .style('font-size', 18);

          svg.append('g')
          .attr('class', key + '-text')
          .style('opacity', 0)
          .attr('transform', 'translate(' + .5 * cfg.width + ',' + .84 * cfg.height + ')')
          .append('text')
              .text('AUC = ' + aucFormat(auc))
              .style('fill', 'white')
              .style('font-size', 18);
      };

      // Draw curves, areas, and text for each
      // true-positive rate in the data
      data.forEach((d, index) => {
          const fpr = JSON.parse(d.fpr);
          const tpr = JSON.parse(d.tpr);

          const auc = calculateArea(fpr, tpr);
          d.auc = auc;

          const points = fpr.map((e, i) => {
              return [e, tpr[i]];
          });

          console.log('drawing the curve for', d.key);
          console.log('color(', index, ')', color(index.toString()));
          drawArea(d.key, points, color(index.toString()));
          drawCurve(d.key, points, color(index.toString()));
          drawAUCText(d.key, d.auc);
      });

      function calculateArea(fpr, tpr) {
          let area = 0.0;
          tpr.forEach((_, i) => {
              if ('undefined' !== typeof fpr[i - 1]) {
                  area += (fpr[i] - fpr[i - 1]) * (tpr[i - 1] + tpr[i]) / 2;
              }
          });
          return area;
      }
  }
}
