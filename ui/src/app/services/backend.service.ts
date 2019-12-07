import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { v4 as uuid } from 'uuid';

import { Results } from '../interfaces';

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  currentJobId;
  previousJobs;
  userData;
  SERVER_URL = 'http://localhost:5000';

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
    return this.http.post<any>(this.SERVER_URL + '/upload/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  startTraining(formData) {
    return this.http.post(this.SERVER_URL + '/train/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  getResults() {
    return this.http.get<Results>(this.SERVER_URL + '/results/' + this.userData.id + '/' + this.currentJobId);
  }

  getModelFeatures(model: string) {
    return this.http.get<string>(this.SERVER_URL + '/features/' + model);
  }

  createModel(formData) {
    return this.http.post(this.SERVER_URL + '/create/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  testPublishedModel(formData, publishName) {
    return this.http.post(this.SERVER_URL + '/test/' + publishName, formData);
  }

  testModel(formData) {
    return this.http.post(this.SERVER_URL + '/test/' + this.userData.id + '/' + this.currentJobId, formData);
  }

  exportCSV() {
    return this.SERVER_URL + '/export/' + this.userData.id + '/' + this.currentJobId;
  }

  exportModel() {
    return this.SERVER_URL + '/export-model/' + this.userData.id + '/' + this.currentJobId;
  }

  exportPMML() {
    return this.SERVER_URL + '/export-pmml/' + this.userData.id + '/' + this.currentJobId;
  }

  exportPublishedModel(publishName) {
    return this.SERVER_URL + '/export-model/' + publishName;
  }

  exportPublishedPMML(publishName) {
    return this.SERVER_URL + '/export-pmml/' + publishName;
  }

  updatePreviousJobs() {
    this.http.get(this.SERVER_URL + '/list-jobs/' + this.userData.id).subscribe(result => {
      this.previousJobs = result;
    });
  }
}
