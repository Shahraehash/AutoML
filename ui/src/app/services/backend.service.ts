import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { v4 as uuid } from 'uuid';

import { ActiveTaskStatus, Results, PendingTasks, PriorJobs, PublishedModels, DataAnalysisReply } from '../interfaces';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  currentJobId: string;
  currentDatasetId: string;
  userData;

  constructor(
    private http: HttpClient,
  ) {
    let userData;
    try {
      userData = JSON.parse(localStorage.getItem('userData'));

      if (userData === null) {
        throw new Error('No user data found');
      }
    } catch (err) {
      userData = {
        id: uuid()
      };
    }

    localStorage.setItem('userData', JSON.stringify(userData));
    this.userData = userData;
  }

  submitData(formData: FormData) {
    return this.http.post<{id: string}>(
      `${environment.apiUrl}/user/${this.userData.id}/datasets`, formData
    ).toPromise().then(reply => {
      this.currentDatasetId = reply.id;
    });
  }

  getDataAnalysis() {
    return this.http.get<DataAnalysisReply>(
      `${environment.apiUrl}/user/${this.userData.id}/datasets/${this.currentDatasetId}/describe`
    );
  }

  createJob() {
    return this.http.post<any>(
      `${environment.apiUrl}/user/${this.userData.id}/jobs`,
      {datasetid: this.currentDatasetId}
    ).toPromise().then(reply => {
      this.currentJobId = reply.id;
    });
  }

  deleteJob(id) {
    return this.http.delete(environment.apiUrl + '/delete/' + this.userData.id + '/' + id);
  }

  startTraining(formData) {
    return this.http.post(`${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/train`, formData);
  }

  getPipelines() {
    return this.http.get(environment.apiUrl + '/pipelines/' + this.userData.id + '/' + this.currentJobId);
  }

  getTaskStatus(id: number) {
    return this.http.get<ActiveTaskStatus>(environment.apiUrl + '/status/' + id);
  }

  cancelTask(id: number) {
    return this.http.delete(environment.apiUrl + '/cancel/' + id);
  }

  getResults() {
    return this.http.get<Results>(
      `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/result`
    );
  }

  getModelFeatures(model: string) {
    return this.http.get<string>(environment.apiUrl + '/features/' + model);
  }

  createModel(formData) {
    return this.http.post(
      `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/refit`,
      formData
    );
  }

  unpublishModel(id: string) {
    return this.http.delete(environment.apiUrl + '/unpublish/' + id);
  }

  testPublishedModel(data, publishName) {
    return this.http.post(environment.apiUrl + '/test/' + publishName, data);
  }

  testModel(data) {
    return this.http.post(
      `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/test`,
      data
    );
  }

  getPendingTasks() {
    return this.http.get<PendingTasks>(environment.apiUrl + '/list-pending/' + this.userData.id);
  }

  getPriorJobs() {
    return this.http.get<PriorJobs[]>(environment.apiUrl + '/list-jobs/' + this.userData.id);
  }

  getPublishedModels() {
    return this.http.get<PublishedModels>(environment.apiUrl + '/list-published/' + this.userData.id);
  }

  exportCSV() {
    return environment.apiUrl + '/export/' + this.userData.id + '/' + this.currentJobId;
  }

  exportModel() {
    return environment.apiUrl + '/export-model/' + this.userData.id + '/' + this.currentJobId;
  }

  exportPMML() {
    return environment.apiUrl + '/export-pmml/' + this.userData.id + '/' + this.currentJobId;
  }

  exportPublishedModel(publishName) {
    return environment.apiUrl + '/export-model/' + publishName;
  }

  exportPublishedPMML(publishName) {
    return environment.apiUrl + '/export-pmml/' + publishName;
  }
}
