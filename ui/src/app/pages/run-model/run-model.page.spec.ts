import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { RunModelPage } from './run-model.page';

describe('RunModelPage', () => {
  let component: RunModelPage;
  let fixture: ComponentFixture<RunModelPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ RunModelPage ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(RunModelPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
