import { Meta, Story } from '@storybook/angular/types-6-0';
import { moduleMetadata } from '@storybook/angular';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatRadioModule } from '@angular/material/radio';
import { MatSelectModule } from '@angular/material/select';

import { FeatureAnalysisComponent } from './feature-analysis.component';
import { IonicModule } from '@ionic/angular';

export default {
  title: 'Components/Feature Details',
  component: FeatureAnalysisComponent,
  decorators: [
    moduleMetadata({
        declarations: [FeatureAnalysisComponent],
        imports: [
            BrowserAnimationsModule,
            CommonModule,
            IonicModule.forRoot(),
            FormsModule,
            ReactiveFormsModule,
            MatFormFieldModule,
            MatSelectModule,
            MatInputModule,
            MatRadioModule
        ]
    })
  ]
} as Meta;

const Template: Story<FeatureAnalysisComponent> = (args) => ({
    props: args,
});

export const BlankFeatureDetail = Template.bind({});
BlankFeatureDetail.args = {};
