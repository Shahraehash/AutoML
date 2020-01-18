import { Component, ElementRef, EventEmitter, Input, Output, OnInit, OnDestroy } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AlertController, LoadingController } from '@ionic/angular';
import { parse } from 'papaparse';
import { Observable, ReplaySubject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import { BackendService } from '../../services/backend.service';
import { DataSets, PublishedModels } from '../../interfaces';

@Component({
  selector: 'app-upload',
  templateUrl: 'upload.component.html',
  styleUrls: ['upload.component.scss'],
})
export class UploadComponent implements OnInit, OnDestroy {
  @Input() isActive: boolean;
  @Output() stepFinished = new EventEmitter();

  destroy$: ReplaySubject<boolean> = new ReplaySubject<boolean>();
  dataSets$: Observable<DataSets[]>;
  publishedModels$: Observable<PublishedModels>;

  labels = [];
  keys = Object.keys;
  uploadForm: FormGroup;

  constructor(
    public backend: BackendService,
    private alertController: AlertController,
    private datePipe: DatePipe,
    private element: ElementRef,
    private formBuilder: FormBuilder,
    private loadingController: LoadingController
  ) {
    this.uploadForm = this.formBuilder.group({
      label_column: ['', Validators.required],
      train: ['', Validators.required],
      test: ['', Validators.required]
    });
  }

  ngOnInit() {
    this.dataSets$ = this.backend.getDataSets().pipe(
      takeUntil(this.destroy$)
    );

    this.publishedModels$ = this.backend.getPublishedModels().pipe(
      takeUntil(this.destroy$)
    );
  }

  ngOnDestroy() {
    this.destroy$.next(true);
    this.destroy$.unsubscribe();
  }

  onSubmit() {
    const formData = new FormData();
    formData.append('train', this.uploadForm.get('train').value);
    formData.append('test', this.uploadForm.get('test').value);
    formData.append('label_column', this.uploadForm.get('label_column').value);

    this.backend.submitData(formData).then(
      () => {
        this.stepFinished.emit({nextStep: 'explore'});
      },
      async () => {
        const alert = await this.alertController.create({
          header: 'Unable to Upload Data',
          message: 'Please make sure the backend is reachable and try again.',
          buttons: ['Dismiss']
        });

        await alert.present();
      }
    );

    return false;
  }

  onFileSelect(event) {
    const errors = this.uploadForm.get(event.target.name).errors;
    if (errors) {
      delete errors.invalidColumn;
    }

    if (event.target.files.length === 1) {
      const file = event.target.files[0];

      parse(file, {
        worker: true,
        step: async (reply, parser) => {
          parser.abort();

          if (event.target.name === 'train') {
            this.labels = reply.data.reverse();
            this.uploadForm.get('test').reset();
          } else {
            if (
              this.labels.length !== reply.data.length ||
              `${this.labels}` !== `${reply.data.reverse()}`
            ) {
              const alert = await this.alertController.create({
                buttons: ['Dismiss'],
                header: 'Data Does Not Match',
                message: 'The columns from the training data does not match the number of columns in the test data.'
              });
              await alert.present();
              this.uploadForm.get(event.target.name).setErrors({
                invalidColumns: true
              });
            }
          }
        }
      });

      this.uploadForm.get(event.target.name).setValue(file);
    }
  }

  exploreDataSet(id) {
    this.backend.currentDatasetId = id;
    this.stepFinished.emit({nextStep: 'explore'});
  }

  async publishedOptions(model) {
    const alert = await this.alertController.create({
      header: `${model.key} Model`,
      subHeader: `This model was published on ${this.datePipe.transform(model.value.date, 'medium')}`,
      message: 'Please select one of the following options:',
      buttons: [
        { text: 'Dismiss' },
        { text: 'Delete', handler: async _ => {
          await alert.dismiss();
          this.deletePublished(model.key);
        } },
        { text: 'Open', handler: _ => setTimeout(() => window.open('/model/' + model.key, '_blank'), 1) }
      ]
    });

    await alert.present();
  }

  async deletePublished(name) {
    const alert = await this.alertController.create({
      buttons: [
        'Dismiss',
        {
          text: 'Delete',
          handler: async () => {
            const loading = await this.loadingController.create({
              message: 'Deleting published model...'
            });
            await loading.present();
            await this.backend.deletePublishedModel(name).toPromise();
            await loading.dismiss();
          }
        }
      ],
      header: 'Are you sure you want to delete?',
      subHeader: 'This cannot be undone.',
      message: 'Are you sure you want to delete this published model?'
    });
    await alert.present();
  }

  reset() {
    this.element.nativeElement.querySelectorAll('input[type="file"]').forEach(node => node.value = '');
    this.uploadForm.reset();
  }
}
