import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { v4 as uuid } from 'uuid';

import { ActiveTaskStatus, Results, PendingTasks, PriorJobs, PublishedModels } from '../interfaces';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  currentJobId;
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

  submitData(formData) {
    this.currentJobId = uuid();
    return this.http.post<any>(environment.apiUrl + '/upload/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  cloneJob(job) {
    this.currentJobId = uuid();
    return this.http.post(environment.apiUrl + '/clone/' + this.userData.id + '/' + job + '/' + this.currentJobId, undefined);
  }

  deleteJob(id) {
    return this.http.delete(environment.apiUrl + '/delete/' + this.userData.id + '/' + id);
  }

  startTraining(formData) {
    return this.http.post(environment.apiUrl + '/train/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  getPipelines() {
    return this.http.get(environment.apiUrl + '/pipelines/' + this.userData.id + '/' + this.currentJobId);
  }

  getDataAnalysis() {
    return this.http.get(environment.apiUrl + '/describe/' + this.userData.id + '/' + this.currentJobId);
  }

  getTaskStatus(id: number) {
    return this.http.get<ActiveTaskStatus>(environment.apiUrl + '/status/' + id);
  }

  cancelTask(id: number) {
    return this.http.delete(environment.apiUrl + '/cancel/' + id);
  }

  getResults() {
    return this.http.get<Results>(environment.apiUrl + '/results/' + this.userData.id + '/' + this.currentJobId);
  }

  getModelFeatures(model: string) {
    return this.http.get<string>(environment.apiUrl + '/features/' + model);
  }

  createModel(formData) {
    return this.http.post(environment.apiUrl + '/create/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  unpublishModel(id: string) {
    return this.http.delete(environment.apiUrl + '/unpublish/' + id);
  }

  testPublishedModel(data, publishName) {
    return this.http.post(environment.apiUrl + '/test/' + publishName, data);
  }

  testModel(data) {
    return this.http.post(environment.apiUrl + '/test/' + this.userData.id + '/' + this.currentJobId, data);
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
