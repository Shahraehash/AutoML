import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { UpdateLicensePage } from './update-license.page';

describe('UpdateLicensePage', () => {
  let component: UpdateLicensePage;
  let fixture: ComponentFixture<UpdateLicensePage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ UpdateLicensePage ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(UpdateLicensePage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
