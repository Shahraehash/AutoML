import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import {
  MatCheckboxModule,
  MatInputModule,
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
    MatStepperModule,
    MatSortModule,
    RouterModule.forChild(routes)
  ],
  declarations: [
    HomePage,
    UploadPage,
    PendingTasksComponent,
    ResultsPage,
    TrainPage,
    RadialDendrogramComponent,
    RocChartComponent,
    UseModelComponent
  ]
})
export class HomePageModule {}
