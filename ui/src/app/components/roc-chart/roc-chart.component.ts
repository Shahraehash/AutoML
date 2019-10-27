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
    @Input() mode: 'mean' | 'test' | 'generalization' | 'reliability';

    private svg;
    private cfg = {
        margin: { top: 30, right: 10, bottom: 70, left: 61 },
        width: 399,
        height: 350,
        tickValues: [0, .1, .25, .5, .75, .9, 1]
    };

    constructor(
        private element: ElementRef
    ) {}

    ngOnInit() {
        this.ngOnChanges();
    }

    ngOnChanges() {
        const format = d3.format('.2');
        const color = scaleOrdinal(schemeCategory10);

        const width = this.cfg.width + this.cfg.margin.left + this.cfg.margin.right;
        const height = this.cfg.height + this.cfg.margin.top + this.cfg.margin.bottom;
        const x = scaleLinear().range([0, this.cfg.width]);
        const y = scaleLinear().range([this.cfg.height, 0]);
        const xAxis = d3Axis.axisBottom(x);
        const yAxis = d3Axis.axisLeft(y);
        xAxis.tickValues(this.cfg.tickValues);
        yAxis.tickValues(this.cfg.tickValues);
        xAxis.tickFormat(format);
        yAxis.tickFormat(format);

        const points = [];
        const sdPoints = [];

        this.data.fpr.forEach((e, i) => {
            points.push([e, this.data.tpr[i]]);

            if (this.data.upper && this.data.lower) {
                sdPoints.push([e, this.data.upper[i], this.data.lower[i]]);
            }
        });

        this.svg = d3.select(this.element.nativeElement).select('svg');
        this.svg.selectAll('*').remove();

        this.svg = this.svg.attr('viewBox', `0 0 ${width} ${height}`).append('g')
            .attr('transform', 'translate(' + this.cfg.margin.left + ',' + this.cfg.margin.top + ')');

        x.domain([0, 1]);
        y.domain([0, 1]);

        this.svg.append('g')
            .attr('class', 'x axis')
            .attr('transform', 'translate(0,' + this.cfg.height + ')')
            .call(xAxis)
            .append('text')
            .attr('x', this.cfg.width / 2)
            .attr('y', 40)
            .style('font-size', '12px')
            .style('text-anchor', 'middle')
            .text(this.mode === 'reliability' ? 'Predicted Probability' : 'False Positive Rate');


        this.svg.append('g')
            .attr('class', 'y axis')
            .call(yAxis)
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -35)
            .attr('x', 0 - this.cfg.height / 2.8)
            .style('font-size', '12px')
            .style('text-anchor', 'left')
            .text(this.mode === 'reliability' ? 'Observed Probability' : 'True Positive Rate');

        this.svg.append('line')
            .attr('class', 'curve')
            .attr('class', this.mode === 'reliability' ? 'ideal' : 'guess')
            .attr('x1', 0)
            .attr('x2', this.cfg.width)
            .attr('y1', this.cfg.height)
            .attr('y2', 0)
            .style('stroke-width', 2)
            .style('stroke-dasharray', 8)
            .style('opacity', .4);

        if (this.mode !== 'reliability') {
            this.drawArea(color('0'), x, y, points);
        }

        if (sdPoints.length) {
            this.drawDeviation(x, y, sdPoints);
        }
        this.drawCurve(color('0'), x, y, points);
        this.drawAUCText(this.data.textElements);
    }

    // A function that returns a line generator
    private curve(x, y, points) {

        const lineGenerator = d3.line()
            // .curve(d3.curveBasis)
            .x((d) => x(d[0]))
            .y((d) => y(d[1]));

        return lineGenerator(points);
    }

    // A function that returns an area generator
    private areaUnderCurve(x, y, height, points) {

        const areaGenerator = d3.area()
            // .curve(d3.curveBasis)
            .x((d) => x(d[0]))
            .y0(height)
            .y1((d) => y(d[1]));

        return areaGenerator(points);
    }

    // Draw the ROC curves
    private drawCurve(stroke, x, y, points) {
        this.svg.append('path')
            .attr('class', 'curve')
            .style('stroke', stroke)
            .attr('d', this.curve(x, y, points));
    }

    // Draw the area under the ROC curves
    private drawArea(fill, x, y, points) {
        this.svg.append('path')
            .attr('class', 'area')
            .attr('d', this.areaUnderCurve(x, y, this.cfg.height, points))
            .style('fill', fill)
            .style('opacity', .2);
    }

    private drawDeviation(x, y, sdPoints) {
        this.svg.append('path')
            .attr('d', d3.area()
                // .curve(d3.curveBasis)
                .x(d => x(d[0]))
                .y0(d => y(d[1]))
                .y1((d: any) => y(d[2]))
                (sdPoints))
            .attr('class', 'deviation')
            .style('fill', 'grey')
            .style('opacity', '.2');
    }

    private drawAUCText(items: any[]) {
        let vAlign = .7;

        items.forEach(item => {
            this.svg.append('g')
                .attr('class', 'text')
                .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + vAlign * this.cfg.height + ')')
                .append('text')
                    .text(item)
                    .attr('class', 'auc-text')
                    .style('font-size', 12);

            vAlign += 0.05;
        });
    }
}
