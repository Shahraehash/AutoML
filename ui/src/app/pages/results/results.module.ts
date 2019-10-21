import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatSortModule } from '@angular/material/sort';
import { MatTableModule } from '@angular/material/table';
import { Routes, RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { ResultsPage } from './results.page';
import { RocChartComponent } from '../../components/roc-chart/roc-chart.component';
import { UseModelComponent } from '../../components/use-model/use-model.component';

const routes: Routes = [
  {
    path: '',
    component: ResultsPage
  }
];

@NgModule({
  entryComponents: [UseModelComponent],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    RouterModule.forChild(routes),
    MatInputModule,
    MatSortModule,
    MatTableModule,
  ],
  declarations: [RocChartComponent, UseModelComponent, ResultsPage]
})
export class ResultsPageModule {}
