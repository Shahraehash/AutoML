import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatStepperModule } from '@angular/material';
import { Routes, RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { HomePage } from './home.page';
import {
  ComponentsModule,
  PendingTasksComponent,
  TextareaModalComponent,
  UseModelComponent
} from '../../components';

const routes: Routes = [
  {
    path: '',
    component: HomePage
  }
];

@NgModule({
  entryComponents: [
    PendingTasksComponent,
    TextareaModalComponent,
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
    HomePage
  ]
})
export class HomePageModule {}
