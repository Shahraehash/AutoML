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
    svg;

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
        const aucFormat = d3.format('.4r');

        const x = scaleLinear().range([0, this.cfg.width]);
        const y = scaleLinear().range([this.cfg.height, 0]);
        const color = scaleOrdinal(schemeCategory10);

        const xAxis = d3Axis.axisBottom(x);
        const yAxis = d3Axis.axisLeft(y);

        xAxis.tickValues(this.cfg.tickValues);
        yAxis.tickValues(this.cfg.tickValues);

        // Apply the format to the ticks we chose
        xAxis.tickFormat(format);
        yAxis.tickFormat(format);

        let upper;
        let lower;
        let fpr;
        let tpr;

        if (this.mode === 'reliability') {
            fpr = JSON.parse(this.data.mpv);
            tpr = JSON.parse(this.data.fop);
        } else {
            fpr = JSON.parse(this.data[this.mode + '_fpr']);
            tpr = JSON.parse(this.data[this.mode + '_tpr']);
        }

        if (this.mode === 'mean') {
            upper = JSON.parse(this.data.mean_upper);
            lower = JSON.parse(this.data.mean_lower);
        }

        const points = [];
        const sdPoints = [];
        const auc = this.calculateArea(fpr, tpr);

        fpr.forEach((e, i) => {
            points.push([e, tpr[i]]);

            if (upper && lower) {
                sdPoints.push([e, upper[i], lower[i]]);
            }
        });

        const width = this.cfg.width + this.cfg.margin.left + this.cfg.margin.right;
        const height = this.cfg.height + this.cfg.margin.top + this.cfg.margin.bottom;

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
            .text(this.mode === 'reliability' ? 'Mean Predicted Value' : 'False Positive Rate');


        this.svg.append('g')
            .attr('class', 'y axis')
            .call(yAxis)
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -35)
            .attr('x', 0 - this.cfg.height / 2.8)
            .style('font-size', '12px')
            .style('text-anchor', 'left')
            .text(this.mode === 'reliability' ? 'Fraction of Positives' : 'True Positive Rate');

        // Draw the random guess line
        this.svg.append('line')
            .attr('class', 'curve')
            .attr('class', 'guess')
            .attr('x1', 0)
            .attr('x2', this.cfg.width)
            .attr('y1', this.cfg.height)
            .attr('y2', 0)
            .style('stroke-width', 2)
            .style('stroke-dasharray', 8)
            .style('opacity', .4);

        if (this.mode !== 'reliability') {
            this.drawArea(this.data.key, color('0'), x, y, points);
        }

        if (sdPoints.length) {
            this.drawDeviation(this.data.key, x, y, sdPoints);
        }
        this.drawCurve(this.data.key, color('0'), x, y, points);
        this.drawAUCText(this.data, auc, aucFormat);
    }

    // A function that returns a line generator
    private curve(x, y, points) {

        const lineGenerator = d3.line()
            .curve(d3.curveBasis)
            .x((d) => x(d[0]))
            .y((d) => y(d[1]));

        return lineGenerator(points);
    }

    // A function that returns an area generator
    private areaUnderCurve(x, y, height, points) {

        const areaGenerator = d3.area()
            .curve(d3.curveBasis)
            .x((d) => x(d[0]))
            .y0(height)
            .y1((d) => y(d[1]));

        return areaGenerator(points);
    }

    // Draw the ROC curves
    private drawCurve(key, stroke, x, y, points) {
        this.svg.append('path')
            .attr('class', 'curve')
            .style('stroke', stroke)
            .attr('d', this.curve(x, y, points))
            .on('mouseover', () => {
                const areaID = '#' + key + '-area';
                this.svg.select(areaID)
                    .style('opacity', .4)
                    .style('visibility', 'initial');
            })
            .on('mouseout', () => {
                const areaID = '#' + key + '-area';
                this.svg.select(areaID)
                    .style('opacity', 0)
                    .style('visibility', 'hidden');
            });
    }

    // Draw the area under the ROC curves
    private drawArea(key, fill, x, y, points) {
        this.svg.append('path')
            .attr('d', this.areaUnderCurve(x, y, this.cfg.height, points))
            .attr('class', 'area')
            .attr('id', key + '-area')
            .style('fill', fill)
            .style('visibility', 'hidden')
            .style('opacity', 0);
    }

    private drawDeviation(key, x, y, sdPoints) {
        this.svg.append('path')
            .attr('d', d3.area()
                .curve(d3.curveBasis)
                .x(d => x(d[0]))
                .y0(d => y(d[1]))
                .y1((d: any) => y(d[2]))
                (sdPoints))
            .attr('class', 'deviation')
            .attr('id', key + '-deviation')
            .style('fill', 'grey')
            .style('opacity', '.2');
    }

    private drawAUCText(item, auc, aucFormat) {
        this.svg.append('g')
            .attr('class', item.key + '-text')
            .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + .70 * this.cfg.height + ')')
            .append('text')
            .text('Estimator: ' + item.estimator)
            .style('fill', 'white')
            .style('font-size', 12);

        this.svg.append('g')
            .attr('class', item.key + '-text')
            .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + .75 * this.cfg.height + ')')
            .append('text')
            .text('Scaler: ' + item.scaler)
            .style('fill', 'white')
            .style('font-size', 12);

        this.svg.append('g')
            .attr('class', item.key + '-text')
            .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + .80 * this.cfg.height + ')')
            .append('text')
            .text('Selector: ' + item.feature_selector)
            .style('fill', 'white')
            .style('font-size', 12);

        this.svg.append('g')
            .attr('class', item.key + '-text')
            .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + .85 * this.cfg.height + ')')
            .append('text')
            .text('Scorer: ' + item.scorer)
            .style('fill', 'white')
            .style('font-size', 12);

        this.svg.append('g')
            .attr('class', item.key + '-text')
            .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + .90 * this.cfg.height + ')')
            .append('text')
            .text('Searcher: ' + item.searcher)
            .style('fill', 'white')
            .style('font-size', 12);


        const text = this.svg.append('g')
            .attr('class', item.key + '-text')
            .attr('transform', 'translate(' + .4 * this.cfg.height + ',' + .95 * this.cfg.height + ')')
            .append('text')
            .style('fill', 'white')
            .style('font-size', 12);

        if (this.mode === 'reliability') {
            text.text('Brier Score: ' + aucFormat(item.brier_score));
        } else {
            text.text('AUC = ' + aucFormat(auc) + (this.mode === 'mean' ? ' ± ' + aucFormat(item.std_auc) : ''));
        }
    }

    private calculateArea(tpr, fpr) {
        let area = 0.0;
        tpr.forEach((_, i) => {
            if ('undefined' !== typeof fpr[i - 1]) {
                area += (fpr[i] - fpr[i - 1]) * (tpr[i - 1] + tpr[i]) / 2;
            }
        });
        return area;
    }
}
