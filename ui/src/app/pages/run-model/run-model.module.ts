import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { RunModelPage } from './run-model.page';
import { UseModelComponent } from '../../components/use-model/use-model.component';

const routes: Routes = [
  {
    path: '',
    component: RunModelPage
  }
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatInputModule,
    IonicModule,
    RouterModule.forChild(routes)
  ],
  declarations: [RunModelPage, UseModelComponent]
})
export class RunModelPageModule {}
