import { Injectable } from '@angular/core';

import { HttpClient } from '@angular/common/http';

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

  startTraining() {
    return this.http.post(this.SERVER_URL + '/train', {});
  }

  getResults() {
    return this.http.get(this.SERVER_URL + '/results');
  }
}
