import { Component, Input } from '@angular/core';
import { GeneralizationResult } from 'src/app/interfaces';
import { PopoverController } from '@ionic/angular';

import { MiloApiService } from '../../services';

@Component({
  selector: 'app-multi-select-menu',
  templateUrl: './multi-select-menu.component.html',
  styleUrls: ['./multi-select-menu.component.scss'],
})
export class MultiSelectMenuComponent {
  @Input() selected: GeneralizationResult[];
  constructor(
    public popoverController: PopoverController,
    private api: MiloApiService
  ) {}

  supportsTandem() {
    if (this.selected.length !== 2) {
      return false;
    }

    const npvModel = this.selected.find(item => item.npv >= .95);
    const ppvModel = this.selected.find(item => item.ppv >= .95);

    return npvModel && ppvModel && npvModel !== ppvModel;
  }

  async starModels() {
    const starred = this.selected.map(item => item.key);
    await this.api.starModels(starred);
    await this.close({starred});
  }

  async unStarModels() {
    const starred = this.selected.map(item => item.key);
    await this.api.unStarModels(starred);
    await this.close({starred});
  }

  async createTandemModel() {
    
  }

  private async close(data) {
    await this.popoverController.dismiss(data);
  }
}
