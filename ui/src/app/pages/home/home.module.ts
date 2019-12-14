import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import {
  MatCheckboxModule,
  MatInputModule,
  MatSelectModule,
  MatStepperModule,
  MatTableModule,
  MatSortModule
} from '@angular/material';
import { Routes, RouterModule } from '@angular/router';

import { IonicModule } from '@ionic/angular';

import { HomePage } from './home.page';
import { ResultsComponent } from '../../components/results/results.component';
import { TrainComponent } from '../../components/train/train.component';
import { UploadComponent } from '../../components/upload/upload.component';
import { PendingTasksComponent } from '../../components/pending-tasks/pending-tasks.component';
import { RadialDendrogramComponent } from '../../components/radial-dendrogram/radial-dendrogram.component';
import { RocChartComponent } from '../../components/roc-chart/roc-chart.component';
import { UseModelComponent } from '../../components/use-model/use-model.component';

const routes: Routes = [
  {
    path: '',
    component: HomePage
  }
];

@NgModule({
  entryComponents: [PendingTasksComponent, UseModelComponent],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    MatCheckboxModule,
    MatInputModule,
    MatTableModule,
    MatSelectModule,
    MatStepperModule,
    MatSortModule,
    RouterModule.forChild(routes)
  ],
  declarations: [
    HomePage,
    UploadComponent,
    PendingTasksComponent,
    ResultsComponent,
    TrainComponent,
    RadialDendrogramComponent,
    RocChartComponent,
    UseModelComponent
  ]
})
export class HomePageModule {}
