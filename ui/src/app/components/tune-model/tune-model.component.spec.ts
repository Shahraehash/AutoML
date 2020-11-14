import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { TuneModelComponent } from './tune-model.component';

describe('TuneModelComponent', () => {
  let component: TuneModelComponent;
  let fixture: ComponentFixture<TuneModelComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TuneModelComponent ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(TuneModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
