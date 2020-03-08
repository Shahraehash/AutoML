import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { AngularFireAuth } from '@angular/fire/auth';
import { v4 as uuid } from 'uuid';

import {
  ActiveTaskStatus,
  DataAnalysisReply,
  DataSets,
  Jobs,
  PendingTasks,
  PublishedModels,
  TestReply,
  Results,
  RefitGeneralization
} from '../../interfaces';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class MiloApiService {
  currentJobId: string;
  currentDatasetId: string;
  localUser: string;

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

      /** If the environment is setup for local user, log out the user */
      if (environment.localUser) {
        this.afAuth.auth.signOut();
      }
    });

    try {
      this.localUser = localStorage.getItem('localUser') || uuid();
    } catch (err) {
      this.localUser = uuid();
    }

    try {
      localStorage.setItem('localUser', this.localUser);
    } catch (err) {}
  }

  async submitData(formData: FormData) {
    return (await this.request<{id: string}>(
      'post',
      `/datasets`,
      formData
    )).toPromise().then(reply => {
      this.currentDatasetId = reply.id;
    });
  }

  getDataAnalysis() {
    return this.request<DataAnalysisReply>(
      'get',
      `/datasets/${this.currentDatasetId}/describe`
    );
  }

  async createJob() {
    return (await this.request<{id: string}>(
      'post',
      `/jobs`,
      {datasetid: this.currentDatasetId}
    )).toPromise().then(reply => {
      this.currentJobId = reply.id;
    });
  }

  deleteJob(id) {
    return this.request('delete', '/jobs/' + id);
  }

  deleteDataset(id) {
    return this.request('delete', '/datasets/' + id);
  }

  startTraining(formData) {
    return this.request(
      'post',
      `/jobs/${this.currentJobId}/train`,
      formData
    );
  }

  getPipelines() {
    return this.request(
      'get',
      `/jobs/${this.currentJobId}/pipelines`
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
      `/jobs/${this.currentJobId}/result`
    );
  }

  getModelFeatures(model: string) {
    return this.request<{features: string; generalization: RefitGeneralization}>('get', `/published/${model}/features`);
  }

  createModel(formData) {
    return this.request(
      'post',
      `/jobs/${this.currentJobId}/refit`,
      formData
    );
  }

  publishModel(name, formData) {
    return this.request(
      'post',
      `/published/${name}`,
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
      `/jobs/${this.currentJobId}/test`,
      data
    );
  }

  getPendingTasks() {
    return this.request<PendingTasks>('get', `/tasks`);
  }

  getDataSets() {
    return this.request<DataSets[]>('get', '/datasets');
  }

  getJobs() {
    return this.request<Jobs[]>('get', '/jobs');
  }

  getPublishedModels() {
    return this.request<PublishedModels>('get', `/published`);
  }

  async exportCSV() {
    return `${environment.apiUrl}/jobs/${this.currentJobId}/export?${await this.getURLAuth()}`;
  }

  async exportModel() {
    return `${environment.apiUrl}/jobs/${this.currentJobId}/export-model?${await this.getURLAuth()}`;
  }

  async exportPMML() {
    return `${environment.apiUrl}/jobs/${this.currentJobId}/export-pmml?${await this.getURLAuth()}`;
  }

  async exportPublishedModel(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-model?${await this.getURLAuth()}`;
  }

  async exportPublishedPMML(publishName) {
    return `${environment.apiUrl}/published/${publishName}/export-pmml?${await this.getURLAuth()}`;
  }

  private async request<T>(method: string, url: string, body?: any) {
    return this.http.request<T>(
      method,
      environment.apiUrl + url,
      {
        body,
        headers: await this.getHttpHeaders()
      }
    );
  }

  private async getHttpHeaders(): Promise<HttpHeaders> {
    return environment.localUser ?
      new HttpHeaders().set('LocalUserID', this.localUser) :
      new HttpHeaders().set('Authorization', `Bearer ${await this.afAuth.auth.currentUser.getIdToken()}`);
  }

  private async getURLAuth(): Promise<string> {
    return environment.localUser ?
      `localUser=${this.localUser}` :
      `currentUser=${await this.afAuth.auth.currentUser.getIdToken()}`;
  }
}
