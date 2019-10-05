import { Component, ElementRef, Input, OnChanges, OnInit } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-radial-dendrogram',
  template: '<svg width="1000" height="1000"></svg>'
})
export class RadialDendrogramComponent implements OnInit, OnChanges {
  @Input() data;
  hierarchy;

  constructor(
    private element: ElementRef,
  ) {}

  ngOnInit() {
    this.ngOnChanges();
  }

  ngOnChanges() {
    if (!this.data) {
      return;
    }

    let parents = [];

    parents.push('pipelines');

    const data = this.data.map(d => {
      const id = 'pipelines.' + d.join('.');

      parents.push(...[
        'pipelines.' + d.slice(0, 1).join('.'),
        'pipelines.' + d.slice(0, 2).join('.'),
        'pipelines.' + d.slice(0, 3).join('.'),
        'pipelines.' + d.slice(0, 4).join('.'),
      ]);
      return {id};
    });

    parents = [...new Set(parents)].map(d => ({id: d}));

    data.push(...parents);
    const svg = d3.select(this.element.nativeElement).select('svg');
    svg.selectAll('*').remove();
    const width = 960;
    const height = 950;
    const g = svg.append('g')
      .attr('transform', 'translate(' + (width / 2 - 15) + ',' + (height / 2 + 25) + ')');

    const stratify = d3.stratify()
      .parentId((d: any) => {
        return d.id.substring(0, d.id.lastIndexOf('.'));
      });

    const tree = d3.cluster()
      .size([360, 390])
      .separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth);

    const root = tree(stratify(data)
        .sort((a, b) => (a.height - b.height) || a.id.localeCompare(b.id)));

    const link = g.selectAll('.link')
      .data(root.descendants().slice(1))
      .enter().append('path')
        .attr('class', 'link')
        .attr('d', d => {
          return 'M' + this.project(d.x, d.y)
              + 'C' + this.project(d.x, (d.y + d.parent.y) / 2)
              + ' ' + this.project(d.parent.x, (d.y + d.parent.y) / 2)
              + ' ' + this.project(d.parent.x, d.parent.y);
        });

    const node = g.selectAll('.node')
      .data(root.descendants())
      .enter().append('g')
        .attr('class', d => 'node' + (d.children ? ' node--internal' : ' node--leaf'))
        .attr('transform', d => 'translate(' + this.project(d.x, d.y) + ')');

    node.append('circle')
        .attr('r', 2.5);

    node.append('text')
        .attr('dy', '.31em')
        .attr('x', d => d.x < 180 === !d.children ? 6 : -6)
        .style('text-anchor', d => d.x < 180 === !d.children ? 'start' : 'end')
        .attr('transform', d => 'rotate(' + (d.x < 180 ? d.x - 90 : d.x + 90) + ')')
        .text(d => d.id.substring(d.id.lastIndexOf('.') + 1));
  }

  private project(x, y) {
    const angle = (x - 90) / 180 * Math.PI;
    const radius = y;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }
}
