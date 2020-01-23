import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';

import { TextareaModalComponent } from './textarea-modal.component';

describe('TextareaModalComponent', () => {
  let component: TextareaModalComponent;
  let fixture: ComponentFixture<TextareaModalComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TextareaModalComponent ],
      imports: [
        FormsModule,
        ReactiveFormsModule,
        IonicModule
      ],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TextareaModalComponent);
    component = fixture.componentInstance;
    component.inputs = [];
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
