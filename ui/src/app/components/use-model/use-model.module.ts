import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { IonicModule } from '@ionic/angular';

import { ModelStatisticsComponent } from '../model-statistics/model-statistics.component';
import { UseModelComponent } from './use-model.component';

@NgModule( {
    declarations: [
        ModelStatisticsComponent,
        UseModelComponent
    ],
    exports: [
        ModelStatisticsComponent,
        UseModelComponent
    ],
    imports: [
        CommonModule,
        IonicModule,
        FormsModule,
        MatInputModule,
        ReactiveFormsModule
    ]
} )
export class UseModelModule {}
