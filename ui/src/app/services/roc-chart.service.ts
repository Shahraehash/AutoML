import { Injectable } from '@angular/core';
import * as d3 from 'd3';
import { schemeCategory10 } from 'd3-scale-chromatic';
import { scaleLinear, scaleOrdinal } from 'd3-scale';
import * as d3Axis from 'd3-axis';

@Injectable({
  providedIn: 'root'
})
export class ROCChartService {
    private cfg = {
        margin: {top: 30, right: 20, bottom: 70, left: 61},
        width: 470,
        height: 450,
        ticks: undefined,
        tickValues: [0, .1, .25, .5, .75, .9, 1],
        fpr: 'fpr',
        tprVariables: [{
            name: 'tpr0',
            label: undefined,
            auc: undefined
        }],
        animate: true
    };

    public create(data, options) {
        const cfg = {...this.cfg, ...options};

        const tprVariables = cfg.tprVariables;
        tprVariables.forEach((d) => {
            if (typeof d.label === 'undefined') {
                d.label = d.name;
            }
        });

        const fpr = cfg.fpr;
        const width = cfg.width;
        const height = cfg.height;
        const animate = cfg.animate;

        const format = d3.format('.2');
        const aucFormat = d3.format('.4r');

        const x = scaleLinear().range([0, width]);
        const y = scaleLinear().range([height, 0]);
        const color = scaleOrdinal(schemeCategory10);

        const xAxis = d3Axis.axisBottom(x);
        const yAxis = d3Axis.axisLeft(y);

        // set the axis ticks based on input parameters,
        // if ticks or tickValues are specified
        if ('undefined' !== typeof cfg.ticks) {
            xAxis.ticks(cfg.ticks);
            yAxis.ticks(cfg.ticks);
        } else if ('undefined' !== typeof cfg.tickValues) {
            xAxis.tickValues(cfg.tickValues);
            yAxis.tickValues(cfg.tickValues);
        } else {
            xAxis.ticks(5);
            yAxis.ticks(5);
        }

        // apply the format to the ticks we chose
        xAxis.tickFormat(format);
        yAxis.tickFormat(format);

        // a function that returns a line generator
        const curve = (input, tpr) => {

            const lineGenerator = d3.line()
                .curve(d3.curveBasis)
                .x((d) => x(d[fpr]))
                .y((d) => y(d[tpr]));

            return lineGenerator(input);
        };

        // a function that returns an area generator
        const areaUnderCurve = (input, tpr) => {

            const areaGenerator = d3.area()
                .x((d) => x(d[fpr]))
                .y0(height)
                .y1((d) => y(d[tpr]));

            return areaGenerator(input);
        };

        const svg = d3.select('#roc')
            .append('svg')
            .attr('width', width + cfg.margin.left + cfg.margin.right)
            .attr('height', height + cfg.margin.top + cfg.margin.bottom)
            .append('g')
                .attr('transform', 'translate(' + cfg.margin.left + ',' + cfg.margin.top + ')');

        x.domain([0, 1]);
        y.domain([0, 1]);

        svg.append('g')
            .attr('class', 'x axis')
            .attr('transform', 'translate(0,' + height + ')')
            .call(xAxis)
            .append('text')
                .attr('x', width / 2)
                .attr('y', 40 )
                .style('text-anchor', 'middle')
                .text('False Positive Rate');


        svg.append('g')
            .attr('class', 'y axis')
            .call(yAxis)
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', -35)
            .attr('x', 0 - height / 2.8)
            .style('font-size', '12px')
            .style('text-anchor', 'left')
            .text('True Positive Rate');

        // draw the ROC curves
        const drawCurve = (input, tpr, stroke) => {
            svg.append('path')
                .attr('class', 'curve')
                .style('stroke', stroke)
                .attr('d', curve(input, tpr))
                .on('mouseover', () => {
                    const areaID = '#' + tpr + 'Area';
                    svg.select(areaID).style('opacity', .4);

                    const aucText = '.' + tpr + 'text';
                    svg.selectAll(aucText).style('opacity', .9);
                })
                .on('mouseout', () => {
                    const areaID = '#' + tpr + 'Area';
                    svg.select(areaID).style('opacity', 0);

                    const aucText = '.' + tpr + 'text';
                    svg.selectAll(aucText).style('opacity', 0);
                });
        };

        // draw the area under the ROC curves
        const drawArea = (input, tpr, fill) => {
            svg.append('path')
            .attr('d', areaUnderCurve(input, tpr))
            .attr('class', 'area')
            .attr('id', tpr + 'Area')
            .style({
                fill,
                opacity: 0
            });
        };

        const drawAUCText = (auc, tpr, label) => {
            svg.append('g')
            .attr('class', tpr + 'text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .5 * width + ',' + .79 * height + ')')
            .append('text')
                .text(label)
                .style({
                    fill: 'white',
                    'font-size': 18
                });

            svg.append('g')
            .attr('class', tpr + 'text')
            .style('opacity', 0)
            .attr('transform', 'translate(' + .5 * width + ',' + .84 * height + ')')
            .append('text')
                .text('AUC = ' + aucFormat(auc))
                .style({
                    fill: 'white',
                    'font-size': 18
                });
        };

        // calculate the area under each curve
        tprVariables.forEach((d) => {
            const tpr = d.name;
            const points = generatePoints(data, fpr, tpr);
            const auc = calculateArea(points);
            d.auc = auc;
        });

        console.log('tprVariables', tprVariables);

        // draw curves, areas, and text for each
        // true-positive rate in the data
        tprVariables.forEach((d, index) => {
            console.log('drawing the curve for', d.label);
            console.log('color(', index, ')', color(index.toString()));
            const tpr = d.name;
            drawArea(data, tpr, color(index.toString()));
            drawCurve(data, tpr, color(index.toString()));
            drawAUCText(d.auc, tpr, d.label);
        });

        ///////////////////////////////////////////////////
        ////// animate through areas for each curve ///////
        ///////////////////////////////////////////////////

        if (animate) {
            // sort tprVariables ascending by AUC
            const tprVariablesAscByAUC = tprVariables.sort((a, b) => {
                return a.auc - b.auc;
            });

            console.log('tprVariablesAscByAUC', tprVariablesAscByAUC);

            for (let i = 0; i < tprVariablesAscByAUC.length; i++) {
            const areaID = '#' + tprVariablesAscByAUC[i].name + 'Area';
            svg.select(areaID)
                .transition()
                .delay(2000 * (i + 1))
                .duration(250)
                .style('opacity', .4)
                .transition()
                .delay(2000 * (i + 2))
                .duration(250)
                .style('opacity', 0);

            const textClass = '.' + tprVariablesAscByAUC[i].name + 'text';
            svg.selectAll(textClass)
                .transition()
                .delay(2000 * (i + 1))
                .duration(250)
                .style('opacity', .9)
                .transition()
                .delay(2000 * (i + 2))
                .duration(250)
                .style('opacity', 0);
            }
        }

        function generatePoints(input, X, Y) {
            const points = [];
            input.forEach((d) => {
                points.push([ Number(d[X]), Number(d[Y]) ]);
            });
            return points;
        }

        // numerical integration
        function calculateArea(points) {
            let area = 0.0;
            const length = points.length;
            if (length <= 2) {
                return area;
            }
            points.forEach((d, i) => {
                const X = 0;
                const Y = 1;

                if ('undefined' !== typeof points[i - 1]) {
                    area += (points[i][X] - points[i - 1][X]) * (points[i - 1][Y] + points[i][Y]) / 2;
                }
            });
            return area;
        }
    }
}
