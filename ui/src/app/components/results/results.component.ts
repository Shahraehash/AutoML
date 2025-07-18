import { SelectionModel } from '@angular/cdk/collections';
import { Component, ViewChild, OnInit, OnDestroy, ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';
import { FormBuilder, FormGroup, FormControl } from '@angular/forms';
import { MatTableDataSource } from '@angular/material/table';
import { MatPaginator } from '@angular/material/paginator';
import { MatSort } from '@angular/material/sort';
import { AlertController, LoadingController, ModalController, ToastController, PopoverController } from '@ionic/angular';
import { saveAs } from 'file-saver';
import * as JSZip from 'jszip';
import * as saveSvgAsPng from 'save-svg-as-png';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import pipelineOptions from '../../data/pipeline.processors.json';
import { MiloApiService } from '../../services/milo-api/milo-api.service';
import { GeneralizationResult, MetaData, RefitGeneralization, Results } from '../../interfaces';
import { MultiSelectMenuComponent } from '../multi-select-menu/multi-select-menu.component';
import { TrainComponent } from '../train/train.component';
import { UseModelComponent } from '../use-model/use-model.component';

@Component({
  selector: 'app-results',
  templateUrl: './results.component.html',
  styleUrls: ['./results.component.scss'],
})
export class ResultsComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();
  private loadingClassData = new Set<string | number>();
  private originalResults: GeneralizationResult[]; // Backup of original macro-averaged data
  
  activeRow = 0;
  data: GeneralizationResult[];
  filterForm: FormGroup;
  loading: HTMLIonLoadingElement;
  rocData;
  metadata: MetaData;
  sortedData: GeneralizationResult[];
  trainingRocData;
  results: MatTableDataSource<GeneralizationResult>;
  selection = new SelectionModel<GeneralizationResult>(true, []);
  starred: string[];
  selectedClass: string | number = 'all';
  classSpecificData: any = {};
  columns: {key: string; class?: string, name: string; number?: boolean, hideOnWidth?: number}[] = [
    {
      key: 'algorithm',
      name: 'Algorithm'
    },
    {
      key: 'roc_auc',
      name: 'ROC AUC',
      hideOnWidth: 375,
      number: true
    },
    {
      key: 'roc_delta',
      name: 'ΔROC AUC',
      hideOnWidth: 1400,
      number: true
    },
    {
      key: 'mcc',
      name: 'MCC',
      hideOnWidth: 375,
      number: true
    },
    {
      key: 'avg_sn_sp',
      name: 'Sn+Sp',
      class: 'overline',
      number: true
    },
    {
      key: 'accuracy',
      name: 'Accuracy',
      hideOnWidth: 400,
      number: true
    },
    {
      key: 'f1',
      name: 'F1',
      hideOnWidth: 1400,
      number: true
    },
    {
      key: 'sensitivity',
      name: 'Sensitivity',
      hideOnWidth: 600,
      number: true
    },
    {
      key: 'specificity',
      name: 'Specificity',
      hideOnWidth: 600,
      number: true
    },
    {
      key: 'npv',
      name: 'NPV',
      hideOnWidth: 600,
      number: true
    },
    {
      key: 'ppv',
      name: 'PPV',
      hideOnWidth: 600,
      number: true
    },
    {
      key: 'brier_score',
      name: 'Brier Score',
      hideOnWidth: 500,
      number: true
    },
    {
      key: 'scaler',
      name: 'Scaler',
      hideOnWidth: 1350
    },
    {
      key: 'feature_selector',
      name: 'Feature Selector',
      hideOnWidth: 1350
    },
    {
      key: 'scorer',
      name: 'Scorer',
      hideOnWidth: 1350
    },
    {
      key: 'searcher',
      name: 'Searcher',
      hideOnWidth: 1350
    },
    {
      key: 'actions',
      name: ''
    }
  ];

  @ViewChild(MatSort) sort: MatSort;
  @ViewChild(MatPaginator) paginator: MatPaginator;

  constructor(
    public api: MiloApiService,
    private alertController: AlertController,
    private formBuilder: FormBuilder,
    private loadingController: LoadingController,
    private modalController: ModalController,
    private toastController: ToastController,
    private popoverController: PopoverController,
    private cdr: ChangeDetectorRef
  ) {
    this.filterForm = this.formBuilder.group({
      query: new FormControl(''),
      group: new FormControl('all')
    });
  }

  async ngOnInit() {
    let data: Results;
    const loading = await this.loadingController.create({
      message: 'Loading results...'
    });
    await loading.present();

    try {
      data = await (await this.api.getResults()).toPromise();
    } catch (err) {
      const alert = await this.alertController.create({
        header: 'Unable to Load Results',
        message: 'Please make sure the backend is reachable and try again.',
        buttons: ['Dismiss']
      });

      await loading.dismiss();
      await alert.present();
    }

    // Store ALL results (including OvR models) for filtering
    this.originalResults = JSON.parse(JSON.stringify(data.results));
    this.metadata = data.metadata;
    
    // Show only main models by default (macro-averaged)
    const mainModels = data.results.filter(result => 
      !result.class_type || result.class_type === 'multiclass' || result.class_type === 'binary'
    );
    
    this.data = mainModels;
    this.results = new MatTableDataSource(mainModels);
    setTimeout(async () => {
      this.results.sort = this.sort;
      this.results.paginator = this.paginator;
      this.results.filterPredicate = this.filter.bind(this);
      await loading.dismiss();
    }, 1);

    this.results.connect().pipe(takeUntil(this.destroy$)).subscribe(d => {
      this.sortedData = d;
    });

    this.updateStarredModels();
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }

  getColumns() {
    return ['select', 'star', ...this.columns.filter(c => !c.hideOnWidth || window.innerWidth > c.hideOnWidth).map(c => c.key)];
  }

  getFilterColumns() {
    return this.columns.filter(i => i.name);
  }

  filter(value, filter) {
    if (filter === 'starred') {
      return this.starred.includes(value.key);
    } else if (filter === 'un-starred') {
      return !this.starred.includes(value.key);
    }

    const group = this.filterForm.get('group').value;
    let dataStr;

    if (group === 'all') {
      dataStr = Object.keys(value).reduce((currentTerm: string, key: string) => {
        return currentTerm + (value as {[key: string]: any})[key] + '◬';
      }, '').toLowerCase();
    } else {
      dataStr = value[group].toLowerCase();
    }


    return dataStr.indexOf(filter.trim().toLowerCase()) !== -1;
  }

  applyFilter() {
    this.results.filter = this.filterForm.get('query').value;
  }

  /** Whether the number of selected elements matches the total number of rows. */
  isAllSelected() {
    const numSelected = this.selection.selected.length;
    const numRows = this.sortedData.length;
    return numSelected === numRows;
  }

  /** Selects all rows if they are not all selected; otherwise clear selection. */
  masterToggle() {
    this.isAllSelected() ?
        this.selection.clear() :
        this.sortedData.forEach(row => this.selection.select(row));
  }

  /** The label for the checkbox on the passed row */
  checkboxLabel(row?: GeneralizationResult): string {
    if (!row) {
      return `${this.isAllSelected() ? 'select' : 'deselect'} all`;
    }
    return `${this.selection.isSelected(row) ? 'deselect' : 'select'} row ${row.key}`;
  }

  async toggleStar(key) {
    await (this.starred.includes(key) ? this.api.unStarModels : this.api.starModels).call(this.api, [key]);
    await this.updateStarredModels();
  }

  toggleStarFilter() {
    switch (this.results.filter) {
      case 'starred':
        this.results.filter = 'un-starred';
        break;
      case 'un-starred':
        this.results.filter = '';
        break;
      default:
        this.results.filter = 'starred';
    }
  }

  parse(object: GeneralizationResult, mode) {
    let fpr;
    let tpr;
    let upper;
    let lower;
    const textElements = [
      'Algorithm: ' + object.algorithm,
      'Scaler: ' + object.scaler,
      'Selector: ' + object.feature_selector,
      'Scorer: ' + object.scorer,
      'Searcher: ' + object.searcher
    ];

    // Check if we should use class-specific data with additional safety checks
    const useClassSpecific = this.selectedClass !== 'all' && 
                             this.classSpecificData && 
                             this.classSpecificData[this.selectedClass] && 
                             this.classSpecificData[this.selectedClass] !== null &&
                             typeof this.classSpecificData[this.selectedClass] === 'object';
    
    if (useClassSpecific) {
      try {
        // Get class-specific data for the current model (object.key)
        const allClassData = this.classSpecificData[this.selectedClass];
        const classData = allClassData[object.key];
        
        if (classData && classData !== null) {
          if (mode === 'generalization' && classData.roc_auc) {
            fpr = classData.roc_auc.fpr || [];
            tpr = classData.roc_auc.tpr || [];
            if (classData.roc_auc.roc_auc !== undefined) {
              textElements.push(`AUC (Class ${this.selectedClass} vs Rest): ` + classData.roc_auc.roc_auc.toFixed(4));
            }
          } else if (mode === 'precision' && classData.precision_recall) {
            fpr = classData.precision_recall.recall || [];
            tpr = classData.precision_recall.precision || [];
            textElements.push(`Class ${this.selectedClass} vs Rest`);
          } else if (mode === 'reliability' && classData.reliability) {
            fpr = classData.reliability.mpv || [];
            tpr = classData.reliability.fop || [];
            if (classData.reliability.brier_score !== undefined) {
              textElements.push(`Brier Score (Class ${this.selectedClass} vs Rest): ` + classData.reliability.brier_score.toFixed(4));
            }
          } else {
            // For modes without class-specific data, fall back to original data
            return this.parseOriginalData(object, mode, textElements);
          }
        } else {
          // No class-specific data for this model, fall back to original data
          return this.parseOriginalData(object, mode, textElements);
        }
      } catch (error) {
        console.error('Error parsing class-specific data:', error);
        // Fall back to original data on any error
        return this.parseOriginalData(object, mode, textElements);
      }
    } else {
      // Use original macro-averaged data
      return this.parseOriginalData(object, mode, textElements);
    }

    return {
      fpr: fpr || [],
      tpr: tpr || [],
      upper,
      lower,
      textElements
    };
  }

  private parseOriginalData(object: GeneralizationResult, mode: string, textElements: string[]) {
    let fpr;
    let tpr;
    let upper;
    let lower;

    if (mode === 'generalization') {
      fpr = JSON.parse(object.generalization_fpr);
      tpr = JSON.parse(object.generalization_tpr);
    } else if (mode === 'precision') {
      fpr = JSON.parse(object.recall);
      tpr = JSON.parse(object.precision);
    } else if (mode === 'reliability') {
      fpr = JSON.parse(object.mpv);
      tpr = JSON.parse(object.fop);
    } else if (mode === 'mean') {
      fpr = JSON.parse(object.mean_fpr);
      tpr = JSON.parse(object.mean_tpr);
      upper = JSON.parse(object.mean_upper);
      lower = JSON.parse(object.mean_lower);
    } else if (mode === 'test') {
      fpr = JSON.parse(object.test_fpr);
      tpr = JSON.parse(object.test_tpr);
    } else {
      return;
    }

    if (mode === 'reliability') {
      const brierScore = typeof object.brier_score === 'string' ? parseFloat(object.brier_score) : object.brier_score;
      textElements.push('Brier Score: ' + brierScore.toFixed(4));
    } else if (mode === 'precision') {
      const f1Score = typeof object.f1 === 'string' ? parseFloat(object.f1) : object.f1;
      textElements.push('F1: ' + f1Score.toFixed(4));
    } else if (mode === 'generalization') {
      const rocAuc = typeof object.roc_auc === 'string' ? parseFloat(object.roc_auc) : object.roc_auc;
      textElements.push('AUC: ' + rocAuc.toFixed(4));
    } else if (mode === 'test') {
      const trainingRocAuc = typeof object.training_roc_auc === 'string' ? parseFloat(object.training_roc_auc) : object.training_roc_auc;
      textElements.push('AUC: ' + trainingRocAuc.toFixed(4));
    }

    return {
      fpr,
      tpr,
      upper,
      lower,
      textElements
    };
  }

  async beginPublish(index: number) {
    const model = this.sortedData[index];

    const alert = await this.alertController.create({
      cssClass: 'wide-alert',
      header: 'Publish Model',
      subHeader: 'Publish your model for standalone use',
      message: `Once published, the model will be available at ${location.origin}/&lt;name&gt;.`,
      inputs: [
        {
          name: 'name',
          type: 'text',
          placeholder: 'Enter the name of your model'
        },
        {
          name: 'threshold',
          type: 'number',
          min: .3,
          max: .7,
          placeholder: 'Decision threshold (.5 default)'
        }
      ],
      buttons: [
        'Dismiss',
        {
          text: 'Publish',
          handler: (data) => {
            if (!data.name || !data.name.match(/^[!#$&-;=?-[\]_a-z~]+$/)) {
              this.showError('Invalid characters detected, please use an alphanumeric name.');
              return false;
            }

            if (data.threshold && (data.threshold < .3 || data.threshold > .7)) {
              this.showError('Invalid threshold detected, please use a value between .3 and .7.');
              return false;
            }

            this.publishModel(model, data.name, data.threshold || .5);
          }
        }
      ]
    });

    alert.present();
  }

  async publishModel(model, name, threshold) {
    await this.presentLoading();
    const formData = new FormData();
    formData.append('key', model.key);
    formData.append('parameters', model.best_params);
    formData.append('features', model.selected_features);
    formData.append('job', this.api.currentJobId);
    formData.append('threshold', threshold);
    formData.append('feature_scores', model.feature_scores);

    (await this.api.publishModel(name, formData)).subscribe(
      async () => {
        const alert = await this.alertController.create({
          buttons: ['Dismiss'],
          cssClass: 'wide-alert',
          header: 'Your model has been published!',
          message: `You may now access your model here:
            <a class='external-link' href='${location.origin}/model/${name}'>${location.origin}/model/${name}</a>`
        });
        await alert.present();
        this.loading.dismiss();
      },
      async () => {
        await this.showError('Unable to publish the model.');
        this.loading.dismiss();
      }
    );
  }

  async launchModel(index: number) {
    await this.presentLoading();
    const formData = new FormData();
    formData.append('key', this.sortedData[index].key);
    formData.append('parameters', this.sortedData[index].best_params);
    formData.append('features', this.sortedData[index].selected_features);

    let scores;
    try {
      scores = JSON.parse(this.sortedData[index].feature_scores);
    } catch(e) {
      scores = {};
    }

    (await this.api.createModel(formData)).subscribe(
      async (reply: {generalization: RefitGeneralization}) => {
        const modal = await this.modalController.create({
          component: UseModelComponent,
          cssClass: 'test-modal',
          componentProps: {
            features: this.sortedData[index].selected_features,
            featureScores: scores,
            generalization: reply.generalization,
            modelKey: this.sortedData[index].key,
            classIndex: this.selectedClass !== 'all' ? this.selectedClass : undefined,
            isMulticlass: this.isMulticlass()
          }
        });
        await modal.present();
        this.loading.dismiss();
      },
      async () => {
        const toast = await this.toastController.create({
          message: 'Unable to create the model.',
          duration: 2000
        });
        toast.present();
        this.loading.dismiss();
      }
    );
  }

  async showDetails() {
    let alert;

    if (this.metadata && this.metadata.fits) {
      const fitDetails = pipelineOptions.estimators.map(estimator => {
        if (!this.metadata.fits[estimator.value]) {
          return '';
        }

        return `
          <ion-item>
            <ion-label>${estimator.label}</ion-label>
            <ion-note slot='end'>${this.metadata.fits[estimator.value]}</ion-note>
          </ion-item>
        `;
      }).join('');

      const message = `
        <ion-list>
          <ion-item>
            <ion-label>Total Models: ${Object.values(this.metadata.fits).reduce((a, b) => a + b, 0)}</ion-label>
            <ion-list>
              ${fitDetails}
            </ion-list>
          </ion-item>
          <ion-item>
            <ion-label>Target Column</ion-label>
            <ion-note slot='end'>${this.metadata.label}</ion-note>
          </ion-item>
          ${this.metadata.train_class_counts ? Object.keys(this.metadata.train_class_counts).map(classKey => {
            const classNumber = classKey.replace('class_', '').replace('_count', '');
            return `
              <ion-item>
                <ion-label>Training Class ${classNumber} Cases</ion-label>
                <ion-note slot='end'>${this.metadata.train_class_counts[classKey]}</ion-note>
              </ion-item>
            `;
          }).join('') : ''}
          ${this.metadata.test_class_counts ? Object.keys(this.metadata.test_class_counts).map(classKey => {
            const classNumber = classKey.replace('class_', '').replace('_count', '');
            return `
              <ion-item>
                <ion-label>Testing (Generalization) Class ${classNumber} Cases</ion-label>
                <ion-note slot='end'>${this.metadata.test_class_counts[classKey]}</ion-note>
              </ion-item>
            `;
          }).join('') : ''}
          <ion-item>
            <ion-label>Cross Validation k-Fold</ion-label>
            <ion-note slot='end'>10</ion-note>
          </ion-item>
        </ion-list>
      `;

      alert = await this.alertController.create({
        cssClass: 'wide-alert',
        buttons: [
          {
            text: 'Download Performance Metrics',
            handler: () => {
              this.api.exportPerformanceCSV().then(url => window.open(url, '_self'));
              return false;
            },
            role: 'secondary'
          },
          {
            text: 'Export Results',
            handler: () => {
              this.api.exportCSV(this.selectedClass).then(url => window.open(url, '_self'));
              return false;
            },
            role: 'secondary'
          },
          'Dismiss'
        ],
        header: 'Analysis Details',
        subHeader: 'Provided below are the details from the model training and validation',
        message
      });
    } else {
      alert = await this.alertController.create({
        buttons: ['Dismiss'],
        header: 'Analysis Details',
        message: 'This run does not contain the metadata needed to display analysis details. This is likely due to an incomplete run.'
      });
    }

    await alert.present();
  }

  async showParameters() {
    const modal = await this.modalController.create({
      cssClass: 'wide-modal',
      component: TrainComponent,
      componentProps: {
        parameters: this.metadata.parameters
      }
    });

    await modal.present();
  }

  async openSelectedOptions(event) {
    const popover = await this.popoverController.create({
      component: MultiSelectMenuComponent,
      componentProps: {
        selected: this.selection.selected
      },
      event
    });
    await popover.present();
    const { data } = await popover.onWillDismiss();
    if (data && data.starred) {
      await this.updateStarredModels();
    }
  }

  async saveCurves() {
    const curves = document.querySelectorAll('app-roc-chart');
    const zip = new JSZip();

    for (let i = 0; i < curves.length; i++) {
      const name = curves[i].getAttribute('mode');
      const image = await saveSvgAsPng.svgAsPngUri(curves[i].querySelector('.roc'), {backgroundColor: 'white'});
      zip.file(name + '.png', image.split(',')[1], {base64: true});
    }

    saveAs(
      await zip.generateAsync({type: 'blob'}),
      `${this.sortedData[this.activeRow].key}_graphs.zip`
    );
  }

  private async updateStarredModels() {
    try {
      this.starred = await this.api.getStarredModels();
    } catch (err) {
      this.starred = [];
    }

    if (['starred', 'un-starred'].includes(this.results?.filter)) {
      this.results.filter = this.results.filter;
    }
  }

  private async presentLoading() {
    this.loading = await this.loadingController.create({
      message: 'Refitting selected model...'
    });
    await this.loading.present();
  }

  isMulticlass(): boolean {
    return this.metadata?.n_classes > 2;
  }

  getClassLabels(): string[] {
    const numClasses = this.metadata?.n_classes;
    if (!numClasses || numClasses <= 2) {
      return [];
    }
    
    // Generate class labels based on number of classes
    const labels = [];
    for (let i = 0; i < numClasses; i++) {
      labels.push(`Class ${i}`);
    }
    return labels;
  }

  onClassChange() {
    console.log('onClassChange called, selectedClass:', this.selectedClass);
    console.log('originalResults length:', this.originalResults?.length);
    
    if (this.selectedClass === 'all') {
      // Show main multiclass models (macro-averaged)
      const mainModels = this.originalResults.filter(result => 
        result.class_type === 'multiclass' || result.class_type === 'binary'
      );
      console.log('Main models found:', mainModels.length);
      console.log('Sample main model:', mainModels[0]);
      this.updateTableData(mainModels);
    } else {
      // Show OvR models for specific class
      console.log('Looking for OvR models with class_type="ovr" and class_index=', this.selectedClass);
      
      // Debug: show all class_type values
      const classTypes = this.originalResults.map(r => r.class_type);
      console.log('All class_type values:', [...new Set(classTypes)]);
      
      // Debug: show all class_index values and their types
      const classIndices = this.originalResults.map(r => r.class_index);
      console.log('All class_index values:', [...new Set(classIndices)]);
      console.log('class_index types:', [...new Set(classIndices.map(v => typeof v))]);
      console.log('selectedClass type:', typeof this.selectedClass);
      
      // Debug: show sample records with class_type and class_index
      const sampleRecords = this.originalResults.slice(0, 5).map(r => ({
        key: r.key,
        class_type: r.class_type,
        class_index: r.class_index,
        class_index_type: typeof r.class_index
      }));
      console.log('Sample records:', sampleRecords);
      
      const classModels = this.originalResults.filter(result => 
        result.class_type === 'ovr' && result.class_index == this.selectedClass
      );
      console.log('OvR models found for class', this.selectedClass, ':', classModels.length);
      if (classModels.length > 0) {
        console.log('Sample OvR model:', classModels[0]);
      }
      this.updateTableData(classModels);
    }
  }

  private async loadClassSpecificData() {
    // Mark as loading to prevent duplicate requests
    this.loadingClassData.add(this.selectedClass);
    
    const loading = await this.loadingController.create({
      message: `Loading Class ${this.selectedClass} data...`
    });
    await loading.present();

    try {
      // Load class-specific data for all models
      const allClassData = {};
      
      // Get all unique model keys from the results
      const modelKeys = this.originalResults.map(result => result.key);
      
      // Load class-specific data for each model
      for (const modelKey of modelKeys) {
        try {
          const result = await (await this.api.getClassSpecificResults(this.selectedClass as number, modelKey))
            .pipe(takeUntil(this.destroy$))
            .toPromise();
          
          allClassData[modelKey] = result;
        } catch (modelErr) {
          console.warn(`Failed to load class data for model ${modelKey}:`, modelErr);
          // Continue with other models even if one fails
          allClassData[modelKey] = null;
        }
      }
      
      this.classSpecificData[this.selectedClass] = allClassData;
    } catch (err) {
      console.error('Error loading class-specific data:', err);
      
      // Mark this class as failed to prevent retrying
      this.classSpecificData[this.selectedClass] = null;
      
      let message = 'Unable to load class-specific data.';
      let resetToAll = false;
      
      // Handle specific error cases
      if (err.error && err.error.code === 'NO_CLASS_RESULTS') {
        message = err.error.message || 'Class-specific results are not available for this job. This job was created before class-specific analysis was implemented.';
        resetToAll = true;
      } else if (err.error && err.error.message) {
        message = err.error.message;
      } else if (err.status === 400) {
        message = 'Class-specific results are not available for this job.';
        resetToAll = true;
      }

      const alert = await this.alertController.create({
        header: 'Class-Specific Data Not Available',
        message: message,
        buttons: [
          {
            text: 'OK',
            handler: () => {
              if (resetToAll) {
                // Reset to 'all' to prevent further errors
                this.selectedClass = 'all';
                this.restoreOriginalData();
              }
            }
          }
        ]
      });
      await alert.present();
    } finally {
      // Always remove from loading set and dismiss loading
      this.loadingClassData.delete(this.selectedClass);
      await loading.dismiss();
    }
  }

  private restoreOriginalData() {
    if (this.originalResults) {
      this.results.data = JSON.parse(JSON.stringify(this.originalResults));
      this.data = JSON.parse(JSON.stringify(this.originalResults));
    }
  }

  private transformTableDataForClass(classIndex: number) {
    if (!this.originalResults || !this.classSpecificData[classIndex]) {
      return;
    }

    const allClassData = this.classSpecificData[classIndex];
    const transformedResults = this.originalResults.map(originalResult => {
      // Create a copy of the original result
      const transformedResult = JSON.parse(JSON.stringify(originalResult));
      
      // Get class-specific data for this specific model
      const modelClassData = allClassData[originalResult.key];
      
      if (modelClassData && modelClassData !== null) {
        try {
          // Update metrics with class-specific values
          if (modelClassData.roc_auc && modelClassData.roc_auc.roc_auc !== undefined) {
            transformedResult.roc_auc = modelClassData.roc_auc.roc_auc;
          }
          
          if (modelClassData.reliability && modelClassData.reliability.brier_score !== undefined) {
            transformedResult.brier_score = modelClassData.reliability.brier_score;
          }

          // Use class-specific ROC delta if available
          if (modelClassData.roc_delta !== undefined && modelClassData.roc_delta !== null) {
            transformedResult.roc_delta = modelClassData.roc_delta;
          } else {
            transformedResult.roc_delta = null;
          }

          // Calculate derived metrics from class-specific data
          const derivedMetrics = this.calculateDerivedMetrics(modelClassData);
          Object.assign(transformedResult, derivedMetrics);

        } catch (error) {
          console.error('Error transforming result for class', classIndex, 'model', originalResult.key, error);
          // Keep original values on error
        }
      }

      return transformedResult;
    });

    // Update the data sources
    this.results.data = transformedResults;
    this.data = transformedResults;
  }

  private calculateDerivedMetrics(classData: any): Partial<GeneralizationResult> {
    const metrics: Partial<GeneralizationResult> = {};

    try {
      // Calculate metrics from ROC curve at optimal threshold (Youden's index)
      if (classData.roc_auc && classData.roc_auc.fpr && classData.roc_auc.tpr) {
        const rocMetrics = this.calculateMetricsFromROC(classData.roc_auc.fpr, classData.roc_auc.tpr);
        Object.assign(metrics, rocMetrics);
      }

      // Calculate F1 from precision-recall curve
      if (classData.precision_recall && classData.precision_recall.precision && classData.precision_recall.recall) {
        const f1 = this.calculateOptimalF1(classData.precision_recall.precision, classData.precision_recall.recall);
        if (f1 !== null) {
          metrics.f1 = f1;
        }
      }

    } catch (error) {
      console.error('Error calculating derived metrics:', error);
    }

    return metrics;
  }

  private calculateMetricsFromROC(fpr: number[], tpr: number[]): any {
    if (!fpr || !tpr || fpr.length !== tpr.length || fpr.length === 0) {
      return {};
    }

    try {
      // Find optimal threshold using Youden's index (TPR - FPR)
      let maxYouden = -1;
      let optimalIndex = 0;
      
      for (let i = 0; i < fpr.length; i++) {
        const youden = tpr[i] - fpr[i];
        if (youden > maxYouden) {
          maxYouden = youden;
          optimalIndex = i;
        }
      }

      const sensitivity = tpr[optimalIndex];
      const specificity = 1 - fpr[optimalIndex];
      const accuracy = (sensitivity + specificity) / 2; // Approximation for balanced classes
      const avg_sn_sp = sensitivity + specificity;

      // Calculate MCC approximation (requires true class distribution for exact calculation)
      // Using a simplified formula based on sensitivity and specificity
      const mcc = Math.sqrt(sensitivity * specificity * (1 - sensitivity) * (1 - specificity)) * 
                  (sensitivity + specificity - 1);

      return {
        sensitivity,
        specificity,
        accuracy,
        avg_sn_sp,
        mcc: isNaN(mcc) ? 0 : mcc
      };
    } catch (error) {
      console.error('Error calculating ROC metrics:', error);
      return {};
    }
  }

  private calculateOptimalF1(precision: number[], recall: number[]): number | null {
    if (!precision || !recall || precision.length !== recall.length || precision.length === 0) {
      return null;
    }

    try {
      let maxF1 = 0;
      
      for (let i = 0; i < precision.length; i++) {
        const p = precision[i];
        const r = recall[i];
        
        if (p + r > 0) {
          const f1 = (2 * p * r) / (p + r);
          if (f1 > maxF1) {
            maxF1 = f1;
          }
        }
      }

      return maxF1;
    } catch (error) {
      console.error('Error calculating F1:', error);
      return null;
    }
  }

  private updateTableData(filteredResults: GeneralizationResult[]) {
    this.data = filteredResults;
    this.results.data = filteredResults;
  }

  private async showError(message: string) {
    const toast = await this.toastController.create({message, duration: 2000});
    return toast.present();
  }
}
