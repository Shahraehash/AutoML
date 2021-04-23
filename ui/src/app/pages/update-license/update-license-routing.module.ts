import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { UpdateLicensePageComponent } from './update-license.page';

const routes: Routes = [
  {
    path: '',
    component: UpdateLicensePageComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class UpdateLicensePageRoutingModule {}
