import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { IonicModule } from '@ionic/angular';

import { UseModelComponent } from './use-model.component';

@NgModule( {
    declarations: [
        UseModelComponent
    ],
    exports: [
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
