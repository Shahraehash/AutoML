import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import {
  MatCheckboxModule,
  MatInputModule,
  MatTableModule,
  MatSortModule
} from '@angular/material';
import { RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { ExploreComponent } from './explore/explore.component';
import { FeatureDetailsComponent } from './feature-details/feature-details.component';
import { PendingTasksComponent } from './pending-tasks/pending-tasks.component';
import { RadialDendrogramComponent } from './radial-dendrogram/radial-dendrogram.component';
import { ResultsComponent } from './results/results.component';
import { RocChartComponent } from './roc-chart/roc-chart.component';
import { TextareaModalComponent } from './textarea-modal/textarea-modal.component';
import { TrainComponent } from './train/train.component';
import { UploadComponent } from './upload/upload.component';
import { UseModelModule } from './use-model/use-model.module';

@NgModule( {
  declarations: [
    ExploreComponent,
    FeatureDetailsComponent,
    PendingTasksComponent,
    RadialDendrogramComponent,
    ResultsComponent,
    RocChartComponent,
    TextareaModalComponent,
    TrainComponent,
    UploadComponent
  ],
  exports: [
    ExploreComponent,
    FeatureDetailsComponent,
    PendingTasksComponent,
    RadialDendrogramComponent,
    ResultsComponent,
    RocChartComponent,
    TextareaModalComponent,
    TrainComponent,
    UploadComponent
  ],
  imports: [
      CommonModule,
      IonicModule,
      FormsModule,
      RouterModule,
      ReactiveFormsModule,
      MatCheckboxModule,
      MatInputModule,
      MatTableModule,
      MatSortModule,
      UseModelModule
  ],
} )
export class ComponentsModule {}
export * from './pending-tasks/pending-tasks.component';
export * from './use-model/use-model.component';
export * from './textarea-modal/textarea-modal.component';
export * from './train/train.component';
