import { Component, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';
import { LoadingController, ModalController, PopoverController, ToastController } from '@ionic/angular';
import { saveAs } from 'file-saver';
import { parse, unparse } from 'papaparse';
import { of } from 'rxjs';
import { finalize } from 'rxjs/operators';

import { MiloApiService } from '../../services/milo-api/milo-api.service';
import { RefitGeneralization, TestReply } from '../../interfaces';
import { TuneModelComponent } from '../tune-model/tune-model.component';

@Component({
  selector: 'app-use-model',
  templateUrl: './use-model.component.html',
  styleUrls: ['./use-model.component.scss'],
})
export class UseModelComponent implements OnInit {
  @Input() features: string;
  @Input() generalization: RefitGeneralization;
  @Input() softGeneralization: RefitGeneralization;
  @Input() hardGeneralization: RefitGeneralization;
  @Input() publishName: string;
  @Input() type: string;
  @Input() threshold = .5;
  parsedFeatures: string[];
  testForm: FormGroup;
  result: TestReply;
  isDragging = false;
  voteType = 'soft';

  constructor(
    public modalController: ModalController,
    private popoverController: PopoverController,
    private api: MiloApiService,
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

  async testModel() {
    switch(this.type) {
      case 'tandem':
        this.testTandemModel();
        break;
      case 'ensemble':
        this.testEnsembleModel();
        break;
      default:
        this.testSingleModel();
    }
  }

  async testSingleModel() {
    let observable;

    if (this.publishName) {
      observable = await this.api.testPublishedModel([this.testForm.get('inputs').value], this.publishName);
    } else {
      observable = await this.api.testModel({
        data: [this.testForm.get('inputs').value],
        threshold: this.threshold
      });
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

  async testTandemModel() {
    this.result = await this.api.testTandemModel({
      data: [this.testForm.get('inputs').value],
      features: this.parsedFeatures
    });
  }

  async testEnsembleModel() {
    this.result = await this.api.testEnsembleModel({
      data: [this.testForm.get('inputs').value],
      features: this.parsedFeatures,
      vote_type: this.voteType
    });
  }

  async generalize(event, type?) {
    event.preventDefault();
    this.endDrag();

    const files = type === 'drop' ? event.dataTransfer.files : event.target.files;

    if (!files.length) {
      return;
    }

    if (files.length > 1) {
      event.target.value = '';
      this.showError('Only one file may be selected at a time.');
      return;
    }

    const loading = await this.loadingController.create({
      message: 'Calculating performance...'
    });
    await loading.present();

    const file = files[0];
    parse(file, {
      dynamicTyping: true,
      worker: true,
      skipEmptyLines: true,
      complete: async reply => {
        event.target.value = '';
        const header = reply.data.shift();

        header.forEach((element, index, arr) => {
          arr[index] = element.toString().trim();
        });

        if (!this.parsedFeatures.every(item => header.includes(item))) {
          await loading.dismiss();
          this.showError('Incoming values do not match expected values. ' +
            'Please check to ensure the required features are included.');
          return;
        }

        const payload = {
          data: reply.data,
          columns: header,
          features: this.parsedFeatures
        };

        try {
          this.generalization = await (
            this.publishName ? this.api.generalizePublished(payload, this.publishName) : this.api.generalize(payload, this.threshold)
          );
        } catch (err) {
          this.showError('Unable to assess model performance. Please ensure the target column is present.');
        }

        await loading.dismiss();
      },
      error: async () => {
        event.target.value = '';

        await loading.dismiss();
        this.showError('Unable to parse the CSV. Please verify a CSV was selected and try again.');
      }
    });
  }

  async batchTest(event, type?) {
    event.preventDefault();
    this.endDrag();

    const files = type === 'drop' ? event.dataTransfer.files : event.target.files;

    if (!files.length) {
      return;
    }

    if (files.length > 1) {
      event.target.value = '';
      this.showError('Only one file may be selected at a time.');
      return;
    }

    const loading = await this.loadingController.create({
      message: 'Calculating probabilities...'
    });
    await loading.present();

    const file = files[0];
    const data = [];
    let header;
    let headerMapping;
    parse(file, {
      dynamicTyping: true,
      worker: true,
      skipEmptyLines: true,
      step: (row, parser) => {
        if (!header) {
          header = row.data;
          header.forEach((element, index, arr) => {
            arr[index] = element.toString().trim();
          });

          if (!this.parsedFeatures.every(item => header.includes(item))) {
            parser.abort();
            return;
          }

          headerMapping = this.parsedFeatures.reduce((result, item) => {
            const index = header.indexOf(item);
            if (index > -1) {
              result.push(index);
            }

            return result;
          }, []);
        } else {
          if (row.data.every(i => typeof i === 'number')) {
            data.push(headerMapping.map(key => row.data[key]));
          }
        }
      },
      complete: async () => {
        event.target.value = '';

        if (!data.length) {
          await loading.dismiss();
          this.showError('Incoming values do not match expected values. ' +
            'Please check to ensure the required features are included.');
          return;
        }

        let observable;

        if (this.publishName) {
          observable = await this.api.testPublishedModel(data, this.publishName);
        } else if (this.type === 'tandem') {
          observable = of(await this.api.testTandemModel({
            data,
            features: this.parsedFeatures
          }));
        } else if (this.type === 'ensemble') {
          observable = of(await this.api.testEnsembleModel({
            data,
            features: this.parsedFeatures,
            vote_type: this.voteType
          }));
        } else {
          observable = await this.api.testModel({
            data,
            threshold: this.threshold
          });
        }

        observable.pipe(
          finalize(() => loading.dismiss())
        ).subscribe(
          (result) => {
            header = [...this.parsedFeatures];
            header.push('prediction', 'probability');
            const mappedData = data.map((i, index) => [...i, result.predicted[index], result.probability[index]]);
            mappedData.unshift(header);
            saveAs(new Blob([unparse(mappedData)], {type: 'text/csv'}), 'results.csv');
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

  async exportBatchTemplate() {
    saveAs(
      new Blob([unparse([this.parsedFeatures])], {type: 'text/csv'}),
      'batch_template.csv'
    );
  }

  async exportModel() {
    window.open(await (this.publishName ? this.api.exportPublishedModel(this.publishName) : this.api.exportModel()), '_self');
  }

  async exportPMML() {
    window.open(await (this.publishName ? this.api.exportPublishedPMML(this.publishName) : this.api.exportPMML()), '_self');
  }

  startDrag(event) {
    event.preventDefault();
    event.stopPropagation();

    this.isDragging = true;
  }

  endDrag() {
    this.isDragging = false;
  }

  async tuneModel(event) {
    const popover = await this.popoverController.create({
      cssClass: 'fit-content',
      component: TuneModelComponent,
      componentProps: {
        threshold: this.type ? undefined : this.threshold,
        voteType: this.type === 'ensemble' ? this.voteType : undefined
      },
      event
    });
    await popover.present();
    const { data } = await popover.onWillDismiss();
    if (data) {
      if (data.threshold) {
        this.threshold = data.threshold;
        this.updateGeneralization();
        if (this.result) {
          this.testModel();
        }
      }

      if (data.voteType) {
        this.voteType = data.voteType;
      }
    }
  }

  private async showError(message: string) {
    const toast = await this.toastController.create({
      message,
      duration: 4000
    });

    await toast.present();
  }

  private async updateGeneralization() {
    const loading = await this.loadingController.create({
      message: 'Calculating performance...'
    });
    await loading.present();
    this.generalization = await this.api.generalize({features: this.parsedFeatures}, this.threshold);
    await loading.dismiss();
  }
}
