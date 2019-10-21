import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { GeneralizationResult } from '../interfaces';

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  SERVER_URL = 'http://localhost:5000';

  constructor(
    private http: HttpClient,
  ) {}

  submitData(formData) {
    return this.http.post<any>(this.SERVER_URL + '/upload', formData);
  }

  startTraining(formData) {
    return this.http.post(this.SERVER_URL + '/train', formData);
  }

  getResults() {
    return this.http.get<GeneralizationResult[]>(this.SERVER_URL + '/results');
  }

  createModel(formData) {
    return this.http.post(this.SERVER_URL + '/create', formData);
  }

  testModel(formData) {
    return this.http.post(this.SERVER_URL + '/test', formData);
  }
}
