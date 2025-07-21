import { Component, ElementRef, Input, OnInit, OnChanges } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-histogram',
  styleUrls: ['./histogram.component.scss'],
  template: '<svg class="histogram" viewBox="0 0 300 240"></svg>'
})
export class HistogramComponent implements OnInit, OnChanges {
  @Input() classData: {[key: string]: [number[], number[]]};
  @Input() classLabelMapping: {[key: string]: string};
  private svg;
  private colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

  constructor(
    private element: ElementRef
  ) {}

  /**
   * Sanitizes class names to be valid CSS selectors
   * Removes/replaces characters that are invalid in CSS class names
   */
  private sanitizeClassName(className: string): string {
    return className
      .replace(/\s+/g, '-')           // Replace spaces with hyphens
      .replace(/[^a-zA-Z0-9\-_]/g, '') // Remove invalid characters, keep alphanumeric, hyphens, underscores
      .replace(/^[0-9]/, 'class-$&')   // Prefix with 'class-' if starts with number
      .toLowerCase();                  // Convert to lowercase for consistency
  }

  /**
   * Gets the display label for a class name
   * Uses class label mapping if available, otherwise falls back to default naming
   */
  private getDisplayLabel(className: string): string {
    // Use class label mapping if available
    if (this.classLabelMapping && this.classLabelMapping[className]) {
      return this.classLabelMapping[className];
    }
    
    // If it's already a custom label, return as is
    if (!className.startsWith('class_')) {
      return className;
    }
    
    // Convert class_0, class_1, etc. to "Class 0", "Class 1", etc.
    return className.replace('class_', 'Class ');
  }

  ngOnInit() {
    this.initializeSvg();
    this.drawHistogram();
  }

  ngOnChanges() {
    if (this.svg) {
      this.svg.selectAll('*').remove();
      this.initializeSvg();
      this.drawHistogram();
    }
  }

  private initializeSvg() {
    this.svg = d3.select(this.element.nativeElement).select('svg');
    const margin = { top: 20, right: 120, bottom: 30, left: 20 };
    const width = 300 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;
    
    this.svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
  }

  private drawHistogram() {

    const margin = { top: 20, right: 120, bottom: 30, left: 20 };
    const width = 300 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;
    
    const g = this.svg.select('g');
    
    // Process data for all classes
    const allClassData = [];
    const classNames = Object.keys(this.classData);
    
    classNames.forEach((className, classIndex) => {
      const [counts, bins] = this.classData[className];
      for (let i = 0; i < counts.length; i++) {
        allClassData.push({
          x: bins[i],
          y: counts[i],
          class: className,
          colorIndex: classIndex % this.colors.length
        });
      }
    });

    if (allClassData.length === 0) {
      return;
    }

    // Set up scales
    const x = d3.scaleLinear()
      .domain(d3.extent(allClassData, d => d.x))
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(allClassData, d => d.y)])
      .range([height, 0]);

    // Add axes
    g.append('g')
      .attr('transform', 'translate(0,' + height + ')')
      .call(d3.axisBottom(x).ticks(5));

    // Group data by class for stacked/grouped bars
    const groupedData = d3.group(allClassData, d => d.x);
    const barWidth = width / groupedData.size;
    
    // Draw bars for each class
    classNames.forEach((className, classIndex) => {
      const classColor = this.colors[classIndex % this.colors.length];
      const classDataPoints = allClassData.filter(d => d.class === className);
      const sanitizedClassName = this.sanitizeClassName(className);
      
      g.selectAll(`.bar-${sanitizedClassName}`)
        .data(classDataPoints)
        .enter().append('rect')
        .attr('class', `bar-${sanitizedClassName}`)
        .attr('x', d => x(d.x) - barWidth / 2 + (classIndex * barWidth / classNames.length))
        .attr('y', d => y(d.y))
        .attr('width', barWidth / classNames.length - 1)
        .attr('height', d => height - y(d.y))
        .attr('fill', classColor)
        .attr('opacity', 0.7);
    });

    // Add legend if multiple classes
    if (classNames.length > 1) {
      const legend = g.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${width + 20}, 10)`);

      classNames.forEach((className, index) => {
        const legendRow = legend.append('g')
          .attr('transform', `translate(0, ${index * 15})`);

        legendRow.append('rect')
          .attr('width', 10)
          .attr('height', 10)
          .attr('fill', this.colors[index % this.colors.length])
          .attr('opacity', 0.7);

        legendRow.append('text')
          .attr('x', 15)
          .attr('y', 8)
          .style('font-size', '10px')
          .text(this.getDisplayLabel(className));
      });
    }
  }
}
