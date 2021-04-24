import { Meta, Story } from '@storybook/angular/types-6-0';
import { moduleMetadata } from '@storybook/angular';
import { CommonModule } from '@angular/common';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { IonicModule } from '@ionic/angular';

import { HistogramComponent } from './histogram.component';

export default {
  title: 'Components/Histogram',
  component: HistogramComponent,
  decorators: [
    moduleMetadata({
      declarations: [HistogramComponent],
      imports: [
        BrowserAnimationsModule,
        CommonModule,
        IonicModule.forRoot()
      ]
    })
  ]
} as Meta;

const Template: Story<HistogramComponent> = (args) => ({
  props: args,
});

export const Histogram = Template.bind({});
Histogram.args = {
  data: [
    [6, 6, 8, 17, 5, 2, 2, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1],
    [28.5, 39.805882352941175, 51.11176470588235, 62.417647058823526, 73.7235294117647, 85.02941176470588, 96.33529411764705, 107.64117647058822, 118.9470588235294, 130.25294117647059, 141.55882352941177, 152.86470588235292, 164.1705882352941, 175.4764705882353, 186.78235294117644, 198.08823529411762, 209.3941176470588, 220.7]
  ]
};
