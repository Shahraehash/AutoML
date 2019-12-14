import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import {
  MatCheckboxModule,
  MatInputModule,
  MatSelectModule,
  MatTableModule,
  MatSortModule
} from '@angular/material';
import { IonicModule } from '@ionic/angular';

import { PendingTasksComponent } from './pending-tasks/pending-tasks.component';
import { RadialDendrogramComponent } from './radial-dendrogram/radial-dendrogram.component';
import { ResultsComponent } from './results/results.component';
import { RocChartComponent } from './roc-chart/roc-chart.component';
import { TextareaModalComponent } from './textarea-modal/textarea-modal.component';
import { TrainComponent } from './train/train.component';
import { UploadComponent } from './upload/upload.component';
import { UseModelComponent } from './use-model/use-model.component';

@NgModule( {
  declarations: [
    PendingTasksComponent,
    RadialDendrogramComponent,
    ResultsComponent,
    RocChartComponent,
    TextareaModalComponent,
    TrainComponent,
    UploadComponent,
    UseModelComponent
  ],
  exports: [
    PendingTasksComponent,
    RadialDendrogramComponent,
    ResultsComponent,
    RocChartComponent,
    TextareaModalComponent,
    TrainComponent,
    UploadComponent,
    UseModelComponent
  ],
  imports: [
      CommonModule,
      IonicModule,
      FormsModule,
      ReactiveFormsModule,
      MatCheckboxModule,
      MatInputModule,
      MatSelectModule,
      MatTableModule,
      MatSortModule
  ],
} )
export class ComponentsModule {}
export * from './pending-tasks/pending-tasks.component';
export * from './use-model/use-model.component';
export * from './textarea-modal/textarea-modal.component';
