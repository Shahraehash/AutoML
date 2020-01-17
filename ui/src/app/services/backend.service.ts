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
    return this.http.delete(environment.apiUrl + '/user/' + this.userData.id + '/jobs/' + id);
  }

  startTraining(formData) {
    return this.http.post(`${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/train`, formData);
  }

  getPipelines() {
    return this.http.get(`${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/pipelines`);
  }

  getTaskStatus(id: number) {
    return this.http.get<ActiveTaskStatus>(`${environment.apiUrl}/tasks/${id}`);
  }

  cancelTask(id) {
    return this.http.delete(`${environment.apiUrl}/tasks/${id}`);
  }

  getResults() {
    return this.http.get<Results>(
      `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/result`
    );
  }

  getModelFeatures(model: string) {
    return this.http.get<string>(`${environment.apiUrl}/published/${model}/features`);
  }

  createModel(formData) {
    return this.http.post(
      `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/refit`,
      formData
    );
  }

  unpublishModel(id: string) {
    return this.http.delete(environment.apiUrl + '/published/' + id);
  }

  testPublishedModel(data, publishName) {
    return this.http.post(`${environment.apiUrl}/published/${publishName}/test`, data);
  }

  testModel(data) {
    return this.http.post(
      `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/test`,
      data
    );
  }

  getPendingTasks() {
    return this.http.get<PendingTasks>(`${environment.apiUrl}/user/${this.userData.id}/tasks`);
  }

  getPriorJobs() {
    return this.http.get<PriorJobs[]>(environment.apiUrl + '/user/' + this.userData.id + '/datasets');
  }

  getPublishedModels() {
    return this.http.get<PublishedModels>(`${environment.apiUrl}/user/${this.userData.id}/published`);
  }

  exportCSV() {
    return `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/export`;
  }

  exportModel() {
    return `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/export-model`;
  }

  exportPMML() {
    return `${environment.apiUrl}/user/${this.userData.id}/jobs/${this.currentJobId}/export-pmml`;
  }

  exportPublishedModel(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-model`;
  }

  exportPublishedPMML(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-pmml`;
  }
}
