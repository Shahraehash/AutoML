import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';
import { RouterModule } from '@angular/router';

import { TrainPage } from './train.page';
import { RadialDendrogramComponent } from '../../components/radial-dendrogram/radial-dendrogram.component';

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
  declarations: [RadialDendrogramComponent, TrainPage]
})
export class TrainPageModule {}
