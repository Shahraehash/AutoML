import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-results',
  templateUrl: './results.page.html',
  styleUrls: ['./results.page.scss'],
})
export class ResultsPage implements OnInit {
  SERVER_URL = 'http://localhost:5000/results';
  results;

  constructor(
    private http: HttpClient,
  ) {}

  ngOnInit() {
    this.http.get(this.SERVER_URL).subscribe(data => this.results = data);
  }

}
