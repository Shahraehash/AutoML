import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { RunModelPage } from './run-model.page';
import { UseModelModule } from '../../components/use-model/use-model.module';

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
    UseModelModule,
    RouterModule.forChild(routes)
  ],
  declarations: [RunModelPage]
})
export class RunModelPageModule {}
