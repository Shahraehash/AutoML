import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { RocChartComponent } from './roc-chart.component';

describe('RocChartComponent', () => {
  let component: RocChartComponent;
  let fixture: ComponentFixture<RocChartComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ RocChartComponent ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(RocChartComponent);
    component = fixture.componentInstance;
    component.data = {
      fpr: [
        0.058292252567053206,
        0.1584972381554116,
        0.2526355895537082,
        0.3599679327052902,
        0.4488937599060371,
        0.5448323613751099,
        0.643644284158716,
        0.7493441369047659,
        0.8364530752726758,
        0.9412140457165812
      ],
      tpr: [
        0.1,
        0.03125,
        0.09615384615384616,
        0.14705882352941177,
        0.23376623376623376,
        0.42424242424242425,
        0.5,
        0.5294117647058824,
        0.5,
        0.4
      ],
      textElements: [
        'Algorithm: logistic regression',
        'Scaler: standard scaler',
        'Selector: all features (no feature selection)',
        'Scorer: accuracy',
        'Searcher: grid search',
        'Brier Score: 0.2089'
      ]
    };
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
