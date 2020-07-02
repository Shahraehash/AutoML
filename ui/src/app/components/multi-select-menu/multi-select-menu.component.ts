import { Component, Input } from '@angular/core';
import { GeneralizationResult } from 'src/app/interfaces';
import { PopoverController, LoadingController, AlertController, ToastController } from '@ionic/angular';

import { MiloApiService } from '../../services';

@Component({
  selector: 'app-multi-select-menu',
  templateUrl: './multi-select-menu.component.html',
  styleUrls: ['./multi-select-menu.component.scss'],
})
export class MultiSelectMenuComponent {
  @Input() selected: GeneralizationResult[];
  constructor(
    private toastController: ToastController,
    private popoverController: PopoverController,
    private loadingController: LoadingController,
    private api: MiloApiService
  ) {}

  supportsTandem() {
    if (this.selected.length !== 2) {
      return false;
    }

    const { npvModel, ppvModel } = this.getNamedTandem();

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
    const loading = await this.loadingController.create();
    await loading.present();

    const { npvModel, ppvModel } = this.getNamedTandem();

    const formData = new FormData();

    formData.append('npv_key', npvModel.key);
    formData.append('npv_parameters', npvModel.best_params);
    formData.append('npv_features', npvModel.selected_features);

    formData.append('ppv_key', ppvModel.key);
    formData.append('ppv_parameters', ppvModel.best_params);
    formData.append('ppv_features', ppvModel.selected_features);

    try {
      await this.api.createTandemModel(formData);
    } catch (err) {
      await this.showError('Unable to create tandem model');
    }

    await loading.dismiss();
    await this.close();
  }

  private async close(data?) {
    await this.popoverController.dismiss(data);
  }

  private getNamedTandem() {
    const npvModel = this.selected.find(item => item.npv >= .95);
    const ppvModel = this.selected.find(item => item.ppv >= .95);
    return {npvModel, ppvModel};
  }

  private async showError(message: string) {
    await (await this.toastController.create({message, duration: 5000})).present();
  }
}
