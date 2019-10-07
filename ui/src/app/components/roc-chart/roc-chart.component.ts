import { Component, ElementRef, Input, OnInit, OnChanges } from '@angular/core';
import * as d3 from 'd3';
import { schemeCategory10 } from 'd3-scale-chromatic';
import { scaleLinear, scaleOrdinal } from 'd3-scale';
import * as d3Axis from 'd3-axis';

@Component({
  selector: 'app-roc-chart',
  template: '<svg class="roc"></svg>'
})
export class RocChartComponent implements OnInit, OnChanges {
  @Input() data;

  private margin = {top: 30, right: 10, bottom: 70, left: 61};
  private width = 470 - this.margin.left - this.margin.right;
  private height = 450 - this.margin.top - this.margin.bottom;
  private rocChartOptions = {
    margin: this.margin,
    width: this.width,
    height: this.height
  };

  constructor(
      private element: ElementRef
  ) {}

  ngOnInit() {
    this.ngOnChanges();
  }

  ngOnChanges() {
      const cfg = {...{
          margin: {top: 30, right: 20, bottom: 70, left: 61},
          width: 470,
          height: 450,
          tickValues: [0, .1, .25, .5, .75, .9, 1]
      }, ...this.rocChartOptions};

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
              .curve(d3.curveBasis)
              .x((d) => x(d[0]))
              .y0(cfg.height)
              .y1((d) => y(d[1]));

          return areaGenerator(points);
      };

      const width = cfg.width + cfg.margin.left + cfg.margin.right;
      const height = cfg.height + cfg.margin.top + cfg.margin.bottom;

      let svg = d3.select(this.element.nativeElement).select('svg');
      svg.selectAll('*').remove();

      svg = svg.attr('viewBox', `0 0 ${width} ${height}`).append('g')
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
              .style('font-size', '12px')
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

      // Draw the random guess line
      svg.append('line')
          .attr('class', 'curve')
          .attr('class', 'guess')
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
                  svg.select(areaID)
                    .style('opacity', .4)
                    .style('visibility', 'initial');

                  const aucText = '.' + key + '-text';
                  svg.selectAll(aucText).style('opacity', .9);
              })
              .on('mouseout', () => {
                  const areaID = '#' + key + '-area';
                  svg.select(areaID)
                    .style('opacity', 0)
                    .style('visibility', 'hidden');

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
            .style('visibility', 'hidden')
            .style('opacity', 0);
      };

      const drawDeviation = (key, points) => {
          svg.append('path')
            .attr('d', d3.area()
                .curve(d3.curveBasis)
                .x(d => x(d[0]))
                .y0(d => y(d[1]))
                .y1((d: any) => y(d[2]))
                (points))
            .attr('class', 'deviation')
            .attr('id', key + '-deviation')
            .style('fill', 'grey')
            .style('opacity', '.2');
      };

      const drawAUCSDText = (item) => {
        svg.append('g')
          .attr('class', item.key + '-sdtext')
          .attr('transform', 'translate(' + .6 * cfg.height + ',' + .65 * cfg.height + ')')
          .append('text')
              .text('AUC: ' + item.trainAuc.toFixed(2))
              .style('fill', 'white')
              .style('font-size', 16);

        svg.append('g')
          .attr('class', item.key + '-sdtext')
          .attr('transform', 'translate(' + .6 * cfg.height + ',' + .70 * cfg.height + ')')
          .append('text')
              .text('SD: <missing>')
              .style('fill', 'white')
              .style('font-size', 16);
      };

      const drawAUCText = (item) => {
          svg.append('g')
            .attr('class', item.key + '-text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .4 * cfg.height + ',' + .70 * cfg.height + ')')
            .append('text')
                .text('Estimator: ' + item.estimator)
                .style('fill', 'white')
                .style('font-size', 12);

          svg.append('g')
            .attr('class', item.key + '-text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .4 * cfg.height + ',' + .75 * cfg.height + ')')
            .append('text')
                .text('Scaler: ' + item.scaler)
                .style('fill', 'white')
                .style('font-size', 12);

          svg.append('g')
            .attr('class', item.key + '-text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .4 * cfg.height + ',' + .80 * cfg.height + ')')
            .append('text')
                .text('Selector: ' + item.feature_selector)
                .style('fill', 'white')
                .style('font-size', 12);

          svg.append('g')
            .attr('class', item.key + '-text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .4 * cfg.height + ',' + .85 * cfg.height + ')')
            .append('text')
                .text('Scorer: ' + item.scorer)
                .style('fill', 'white')
                .style('font-size', 12);

          svg.append('g')
            .attr('class', item.key + '-text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .4 * cfg.height + ',' + .90 * cfg.height + ')')
            .append('text')
                .text('Searcher: ' + item.searcher)
                .style('fill', 'white')
                .style('font-size', 12);

          svg.append('g')
            .attr('class', item.key + '-text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .4 * cfg.height + ',' + .95 * cfg.height + ')')
            .append('text')
                .text('AUC = ' + aucFormat(item.auc))
                .style('fill', 'white')
                .style('font-size', 12);
      };

      if (Array.isArray(this.data)) {

        // Draw curves, areas, and text for each
        // true-positive rate in the data
        this.data.forEach((d, index) => {
            const fpr = JSON.parse(d.test_fpr);
            const tpr = JSON.parse(d.test_tpr);

            const auc = calculateArea(fpr, tpr);
            d.trainAuc = auc;

            const points = fpr.map((e, i) => {
                return [e, tpr[i]];
            });

            drawArea(d.key, points, color(index.toString()));
            drawCurve(d.key, points, color(index.toString()));
            drawAUCText(d);
        });
      } else {
        const fpr = JSON.parse(this.data.mean_fpr);
        const tpr = JSON.parse(this.data.mean_tpr);
        const upper = JSON.parse(this.data.tprs_upper);
        const lower = JSON.parse(this.data.tprs_lower);
        const auc = calculateArea(fpr, tpr);
        this.data.trainAuc = auc;

        const points = [];
        const sdPoints = [];

        fpr.forEach((e, i) => {
            points.push([e, tpr[i]]);
            sdPoints.push([e, upper[i], lower[i]]);
        });

        drawCurve(this.data.key, points, color('0'));
        drawDeviation(this.data.key, sdPoints);
        drawAUCSDText(this.data);
      }

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
