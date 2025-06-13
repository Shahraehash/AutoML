import { CdkVirtualScrollViewport } from '@angular/cdk/scrolling';
import { AfterViewInit, Component, EventEmitter, Input, OnInit, Output, ViewChild } from '@angular/core';

import { DataAnalysis } from '../../interfaces';

@Component({
  selector: 'app-feature-details',
  templateUrl: './feature-details.component.html',
  styleUrls: ['./feature-details.component.scss'],
})
export class FeatureDetailsComponent implements OnInit, AfterViewInit {
  @Input() data: DataAnalysis;
  @Input() label: string;
  @Input() type: string;
  @ViewChild(CdkVirtualScrollViewport) viewPort: CdkVirtualScrollViewport;
  @Output() contentScroll = new EventEmitter();
  
  features: string[];

  ngOnInit() {
    this.features = Object.keys(this.data.summary).filter(item => item !== this.label);
  }

  ngAfterViewInit(): void {
    this.viewPort.elementScrolled().subscribe(() => {
      this.contentScroll.emit(this.viewPort.measureScrollOffset('top'));
    });
  }

  updateScroll(position: number) {
    if (this.viewPort && typeof position === 'number') {
      this.viewPort.scrollTo({top: position});
    }
  }

  getClassHistograms(feature: string): {[key: string]: [number[], number[]]} {
    const result = {};
    for (const [className, classData] of Object.entries(this.data.histogram.by_class)) {
      if (classData[feature]) {
        result[className] = classData[feature];
      }
    }
    return result;
  }
}
