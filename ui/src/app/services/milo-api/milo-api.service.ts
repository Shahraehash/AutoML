import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { AngularFireAuth } from '@angular/fire/auth';

import {
  ActiveTaskStatus,
  DataAnalysisReply,
  DataSets,
  Jobs,
  PendingTasks,
  PublishedModels,
  TestReply,
  Results
} from '../../interfaces';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class MiloApiService {
  currentJobId: string;
  currentDatasetId: string;

  constructor(
    private afAuth: AngularFireAuth,
    private http: HttpClient
  ) {
    this.afAuth.authState.subscribe(user => {
      if (!user) {
        this.currentDatasetId = undefined;
        this.currentJobId = undefined;
        return;
      }
    });
  }

  async submitData(formData: FormData) {
    return (await this.request<{id: string}>(
      'post',
      `/user/${this.afAuth.auth.currentUser.uid}/datasets`,
      formData
    )).toPromise().then(reply => {
      this.currentDatasetId = reply.id;
    });
  }

  getDataAnalysis() {
    return this.request<DataAnalysisReply>(
      'get',
      `/user/${this.afAuth.auth.currentUser.uid}/datasets/${this.currentDatasetId}/describe`
    );
  }

  async createJob() {
    return (await this.request<{id: string}>(
      'post',
      `/user/${this.afAuth.auth.currentUser.uid}/jobs`,
      {datasetid: this.currentDatasetId}
    )).toPromise().then(reply => {
      this.currentJobId = reply.id;
    });
  }

  deleteJob(id) {
    return this.request('delete', '/user/' + this.afAuth.auth.currentUser.uid + '/jobs/' + id);
  }

  deleteDataset(id) {
    return this.request('delete', '/user/' + this.afAuth.auth.currentUser.uid + '/datasets/' + id);
  }

  startTraining(formData) {
    return this.request(
      'post',
      `/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/train`,
      formData
    );
  }

  getPipelines() {
    return this.request(
      'get',
      `/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/pipelines`
    );
  }

  getTaskStatus(id: number) {
    return this.request<ActiveTaskStatus>('get', `/tasks/${id}`);
  }

  cancelTask(id) {
    return this.request('delete', `/tasks/${id}`);
  }

  getResults() {
    return this.request<Results>(
      'get',
      `/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/result`
    );
  }

  getModelFeatures(model: string) {
    return this.request<string>('get', `/published/${model}/features`);
  }

  createModel(formData) {
    return this.request(
      'post',
      `/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/refit`,
      formData
    );
  }

  deletePublishedModel(name: string) {
    return this.request('delete', '/published/' + name);
  }

  testPublishedModel(data, publishName) {
    return this.request<TestReply>(
      'post',
      `/published/${publishName}/test`,
      data
    );
  }

  testModel(data) {
    return this.request<TestReply>(
      'post',
      `/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/test`,
      data
    );
  }

  getPendingTasks() {
    return this.request<PendingTasks>('get', `/user/${this.afAuth.auth.currentUser.uid}/tasks`);
  }

  getDataSets() {
    return this.request<DataSets[]>('get', '/user/' + this.afAuth.auth.currentUser.uid + '/datasets');
  }

  getJobs() {
    return this.request<Jobs[]>('get', '/user/' + this.afAuth.auth.currentUser.uid + '/jobs');
  }

  getPublishedModels() {
    return this.request<PublishedModels>('get', `/user/${this.afAuth.auth.currentUser.uid}/published`);
  }

  exportCSV() {
    return `${environment.apiUrl}/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/export`;
  }

  exportModel() {
    return `${environment.apiUrl}/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/export-model`;
  }

  exportPMML() {
    return `${environment.apiUrl}/user/${this.afAuth.auth.currentUser.uid}/jobs/${this.currentJobId}/export-pmml`;
  }

  exportPublishedModel(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-model`;
  }

  exportPublishedPMML(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-pmml`;
  }

  private async request<T>(method: string, url: string, body?: any) {
    const request = this.http.request<T>(
      method,
      environment.apiUrl + url,
      {
        body,
        headers: await this.getHttpHeaders()
      }
    );

    return request;
  }

  private async getHttpHeaders(): Promise<HttpHeaders> {
    return new HttpHeaders().set('Authorization', `Bearer ${await this.afAuth.auth.currentUser.getIdToken()}`);
  }
}
