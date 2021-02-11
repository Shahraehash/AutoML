import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { UpdateLicensePage } from './update-license.page';

const routes: Routes = [
  {
    path: '',
    component: UpdateLicensePage
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class UpdateLicensePageRoutingModule {}
