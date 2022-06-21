import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatStepperModule } from '@angular/material/stepper';
import { Routes, RouterModule } from '@angular/router';
import { IonicModule } from '@ionic/angular';

import { SearchPageComponent } from './search.page';
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
    component: SearchPageComponent
  }
];

@NgModule({
    imports: [
        CommonModule,
        FormsModule,
        ReactiveFormsModule,
        IonicModule,
        MatIconModule,
        MatStepperModule,
        ComponentsModule,
        RouterModule.forChild(routes)
    ],
    declarations: [
        SearchPageComponent
    ]
})
export class SearchPageModule {}
