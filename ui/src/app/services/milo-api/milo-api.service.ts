import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AngularFireAuth } from '@angular/fire/auth';
import { v4 as uuid } from 'uuid';

import {
  ActiveTaskStatus,
  DataAnalysisReply,
  DataSets,
  Jobs,
  PendingTasks,
  PublishedModels,
  Results
} from '../../interfaces';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class MiloApiService {
  currentJobId: string;
  currentDatasetId: string;
  currentUser: string;

  constructor(
    private afAuth: AngularFireAuth,
    private http: HttpClient
  ) {
    this.afAuth.authState.subscribe(user => {
      if (!user) {
        this.currentDatasetId = undefined;
        this.currentJobId = undefined;
        this.currentUser = undefined;
        return;
      }

      this.currentUser = user.uid;
    });
  }

  submitData(formData: FormData) {
    return this.http.post<{id: string}>(
      `${environment.apiUrl}/user/${this.currentUser}/datasets`, formData
    ).toPromise().then(reply => {
      this.currentDatasetId = reply.id;
    });
  }

  getDataAnalysis() {
    return this.http.get<DataAnalysisReply>(
      `${environment.apiUrl}/user/${this.currentUser}/datasets/${this.currentDatasetId}/describe`
    );
  }

  createJob() {
    return this.http.post<any>(
      `${environment.apiUrl}/user/${this.currentUser}/jobs`,
      {datasetid: this.currentDatasetId}
    ).toPromise().then(reply => {
      this.currentJobId = reply.id;
    });
  }

  deleteJob(id) {
    return this.http.delete(environment.apiUrl + '/user/' + this.currentUser + '/jobs/' + id);
  }

  deleteDataset(id) {
    return this.http.delete(environment.apiUrl + '/user/' + this.currentUser + '/datasets/' + id);
  }

  startTraining(formData) {
    return this.http.post(`${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/train`, formData);
  }

  getPipelines() {
    return this.http.get(`${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/pipelines`);
  }

  getTaskStatus(id: number) {
    return this.http.get<ActiveTaskStatus>(`${environment.apiUrl}/tasks/${id}`);
  }

  cancelTask(id) {
    return this.http.delete(`${environment.apiUrl}/tasks/${id}`);
  }

  getResults() {
    return this.http.get<Results>(
      `${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/result`
    );
  }

  getModelFeatures(model: string) {
    return this.http.get<string>(`${environment.apiUrl}/published/${model}/features`);
  }

  createModel(formData) {
    return this.http.post(
      `${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/refit`,
      formData
    );
  }

  deletePublishedModel(name: string) {
    return this.http.delete(environment.apiUrl + '/published/' + name);
  }

  testPublishedModel(data, publishName) {
    return this.http.post(`${environment.apiUrl}/published/${publishName}/test`, data);
  }

  testModel(data) {
    return this.http.post(
      `${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/test`,
      data
    );
  }

  getPendingTasks() {
    return this.http.get<PendingTasks>(`${environment.apiUrl}/user/${this.currentUser}/tasks`);
  }

  getDataSets() {
    return this.http.get<DataSets[]>(environment.apiUrl + '/user/' + this.currentUser + '/datasets');
  }

  getJobs() {
    return this.http.get<Jobs[]>(environment.apiUrl + '/user/' + this.currentUser + '/jobs');
  }

  getPublishedModels() {
    return this.http.get<PublishedModels>(`${environment.apiUrl}/user/${this.currentUser}/published`);
  }

  exportCSV() {
    return `${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/export`;
  }

  exportModel() {
    return `${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/export-model`;
  }

  exportPMML() {
    return `${environment.apiUrl}/user/${this.currentUser}/jobs/${this.currentJobId}/export-pmml`;
  }

  exportPublishedModel(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-model`;
  }

  exportPublishedPMML(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-pmml`;
  }
}
