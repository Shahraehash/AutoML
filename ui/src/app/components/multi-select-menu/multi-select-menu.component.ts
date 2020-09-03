import { Component, Input } from '@angular/core';
import { PopoverController, LoadingController, ToastController, ModalController } from '@ionic/angular';

import { environment } from '../../../environments/environment';
import { MiloApiService } from '../../services';
import { GeneralizationResult, RefitGeneralization } from '../../interfaces';
import { UseModelComponent } from '../use-model/use-model.component';

@Component({
  selector: 'app-multi-select-menu',
  templateUrl: './multi-select-menu.component.html',
  styleUrls: ['./multi-select-menu.component.scss'],
})
export class MultiSelectMenuComponent {
  @Input() selected: GeneralizationResult[];
  showAdvanced = !environment.production;

  constructor(
    private toastController: ToastController,
    private popoverController: PopoverController,
    private loadingController: LoadingController,
    private modalController: ModalController,
    private api: MiloApiService
  ) {}

  supportsTandem() {
    if (this.selected.length !== 2) {
      return false;
    }

    const { npvModel, ppvModel } = this.getNamedTandem();

    return npvModel && ppvModel && npvModel !== ppvModel;
  }

  supportsEnsemble() {
    return this.selected.length % 2 && this.selected.length > 2 && this.selected.length < 6;
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
      return;
    } finally {
      await loading.dismiss();
      await this.close();
    }

    const features = JSON.stringify(
      [...new Set(
        this.parseFeatures(npvModel.selected_features).concat(this.parseFeatures(ppvModel.selected_features))
      )]
    );

    const modal = await this.modalController.create({
      component: UseModelComponent,
      cssClass: 'test-modal',
      componentProps: {features, type: 'tandem'}
    });

    await modal.present();
  }

  async createEnsembleModel() {
    const loading = await this.loadingController.create();
    await loading.present();

    const formData = new FormData();

    let x = 0;
    for (const model of this.selected) {
      formData.append(`model${x}_key`, model.key);
      formData.append(`model${x}_parameters`, model.best_params);
      formData.append(`model${x}_features`, model.selected_features);
      x++;
    }

    formData.append('total_models', x.toString());

    let reply: {hard_generalization: RefitGeneralization, soft_generalization: RefitGeneralization};
    try {
      reply = await this.api.createEnsembleModel(formData);
    } catch (err) {
      await this.showError('Unable to create ensemble model');
      return;
    } finally {
      await loading.dismiss();
      await this.close();
    }

    const features = JSON.stringify(
      [...new Set(
        this.selected.map(model => this.parseFeatures(model.selected_features)).flat()
      )]
    );

    const modal = await this.modalController.create({
      component: UseModelComponent,
      cssClass: 'test-modal',
      componentProps: {
        features,
        type: 'ensemble',
        softGeneralization: reply.soft_generalization,
        hardGeneralization: reply.hard_generalization
      }
    });

    await modal.present();
  }

  private async close(data?) {
    await this.popoverController.dismiss(data);
  }

  private getNamedTandem() {
    const npvModel = this.selected.find(item => item.npv >= .95);
    const ppvModel = this.selected.find(item => item.ppv >= .95);
    return {npvModel, ppvModel};
  }

  private parseFeatures(features) {
    return JSON.parse(features.replace(/'/g, '"'));
  }

  private async showError(message: string) {
    await (await this.toastController.create({message, duration: 5000})).present();
  }
}
