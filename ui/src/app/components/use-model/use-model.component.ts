import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { LoadingController, ToastController } from '@ionic/angular';
import { parse, unparse } from 'papaparse';

import { BackendService } from '../../services/backend.service';
import { finalize } from 'rxjs/operators';

@Component({
  selector: 'app-use-model',
  templateUrl: './use-model.component.html',
  styleUrls: ['./use-model.component.scss'],
})
export class UseModelComponent implements OnInit {
  @Input() features: string;
  @Input() publishName: string;
  parsedFeatures: string[];
  testForm: FormGroup;
  result;

  constructor(
    private backend: BackendService,
    private formBuilder: FormBuilder,
    private loadingController: LoadingController,
    private toastController: ToastController
  ) {}

  ngOnInit() {
    this.parsedFeatures = JSON.parse(this.features.replace(/'/g, '"'));

    this.testForm = this.formBuilder.group({
      inputs: this.formBuilder.array(
        new Array(this.parsedFeatures.length).fill(['', Validators.required])
      )
    });
  }

  testModel() {
    let observable;

    if (this.publishName) {
      observable = this.backend.testPublishedModel([this.testForm.get('inputs').value], this.publishName);
    } else {
      observable = this.backend.testModel([this.testForm.get('inputs').value]);
    }

    observable.subscribe(
      (result) => {
        this.result = result;
      },
      () => {
        this.result = undefined;
      }
    );
  }

  async batchTest(event) {
    if (!event.target.files.length) {
      return;
    }

    if (event.target.files.length > 1) {
      event.target.value = '';
      this.showError('Only one file may be selected at a time.');
      return;
    }

    const loading = await this.loadingController.create({
      message: 'Calculating Probabilities...'
    });
    await loading.present();

    const file = event.target.files[0];
    parse(file, {
      worker: true,
      complete: async reply => {
        let observable;
        const header = reply.data.shift();

        event.target.value = '';

        if (JSON.stringify(header) !== JSON.stringify(this.parsedFeatures)) {
          await loading.dismiss();
          this.showError('Incoming values do not match expected values. ' +
            'Please check the columns are in the right order and the correct number of columns exists.');
          return;
        }

        if (this.publishName) {
          observable = this.backend.testPublishedModel(reply.data, this.publishName);
        } else {
          observable = this.backend.testModel(reply.data);
        }

        observable.pipe(
          finalize(() => loading.dismiss())
        ).subscribe(
          (result) => {
            header.push('predicated', 'probability');
            const data = reply.data.map((i, index) => [...i, result.predicted[index], result.probability[index]]);
            data.unshift(header);
            this.saveCSV(unparse(data));
          },
          () => {
            this.showError('Unable to test the data, please validate the data and try again.');
          }
        );
      },
      error: async () => {
        event.target.value = '';

        await loading.dismiss();
        this.showError('Unable to parse the CSV. Please verify a CSV was selected and try again.');
      }
    });
  }

  exportModel() {
    window.open(this.publishName ? this.backend.exportPublishedModel(this.publishName) : this.backend.exportModel(), '_self');
  }

  exportPMML() {
    window.open(this.publishName ? this.backend.exportPublishedPMML(this.publishName) : this.backend.exportPMML(), '_self');
  }

  private async showError(message: string) {
    const toast = await this.toastController.create({
      message,
      duration: 4000
    });

    await toast.present();
  }

  private saveCSV(csvString) {
    const blob = new Blob([csvString]);
    if (window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveBlob(blob, 'results.csv');
    } else {
        const a = window.document.createElement('a');
        a.href = window.URL.createObjectURL(blob);
        a.download = 'results.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
  }
}
