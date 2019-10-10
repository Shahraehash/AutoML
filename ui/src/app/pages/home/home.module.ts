import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import {
  MatCheckboxModule,
  MatStepperModule,
  MatTableModule,
  MatSortModule
} from '@angular/material';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { HomePage } from './home.page';
import { ResultsPage } from '../../components/results/results.page';
import { TrainPage } from '../../components/train/train.page';
import { UploadPage } from '../../components/upload/upload.page';
import { RadialDendrogramComponent } from '../../components/radial-dendrogram/radial-dendrogram.component';
import { RocChartComponent } from '../../components/roc-chart/roc-chart.component';

const routes: Routes = [
  {
    path: '',
    component: HomePage
  }
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    MatCheckboxModule,
    MatTableModule,
    MatStepperModule,
    MatSortModule,
    RouterModule.forChild(routes)
  ],
  declarations: [HomePage, UploadPage, ResultsPage, TrainPage, RadialDendrogramComponent, RocChartComponent]
})
export class HomePageModule {}
