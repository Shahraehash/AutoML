import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';
import { RouterModule } from '@angular/router';

import { TrainPage } from './train.page';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    MatCheckboxModule,
    RouterModule.forChild([
      {
        path: '',
        component: TrainPage
      }
    ])
  ],
  declarations: [TrainPage]
})
export class TrainPageModule {}
