import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatStepperModule } from '@angular/material';
import { Routes, RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { SearchPage } from './search.page';
import {
  ComponentsModule,
  PendingTasksComponent,
  TextareaModalComponent,
  TrainComponent,
  UseModelComponent
} from '../../components';

const routes: Routes = [
  {
    path: '',
    component: SearchPage
  }
];

@NgModule({
  entryComponents: [
    PendingTasksComponent,
    TextareaModalComponent,
    TrainComponent,
    UseModelComponent
  ],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    MatStepperModule,
    ComponentsModule,
    RouterModule.forChild(routes)
  ],
  declarations: [
    SearchPage
  ]
})
export class SearchPageModule {}
