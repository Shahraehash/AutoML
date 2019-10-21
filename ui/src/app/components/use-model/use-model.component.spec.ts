import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { UseModelComponent } from './use-model.component';

describe('UseModelComponent', () => {
  let component: UseModelComponent;
  let fixture: ComponentFixture<UseModelComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ UseModelComponent ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(UseModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
