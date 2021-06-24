import { Component, EventEmitter, Input, OnInit, Output, ViewChild } from '@angular/core';
import { IonContent } from '@ionic/angular';

import { DataAnalysis } from '../../interfaces';

@Component({
  selector: 'app-feature-details',
  templateUrl: './feature-details.component.html',
  styleUrls: ['./feature-details.component.scss'],
})
export class FeatureDetailsComponent implements OnInit {
  @Input() data: DataAnalysis;
  @Input() label: string;
  @Input() type: string;
  @Output() contentScroll = new EventEmitter();
  
  features: string[];

  @ViewChild('content') private content: IonContent;
  
  ngOnInit() {
    this.features = Object.keys(this.data.summary).filter(item => item !== this.label);
  }

  updateScroll(position: number) {
    if (this.content && typeof position === 'number') {
      this.content.scrollToPoint(0, position, 0);
    }
  }
}
