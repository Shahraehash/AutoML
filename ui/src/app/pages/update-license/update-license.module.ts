import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { IonicModule } from '@ionic/angular';

import { UpdateLicensePageRoutingModule } from './update-license-routing.module';

import { UpdateLicensePageComponent } from './update-license.page';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    UpdateLicensePageRoutingModule
  ],
  declarations: [UpdateLicensePageComponent]
})
export class UpdateLicensePageModule {}
