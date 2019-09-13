import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-train',
  templateUrl: 'train.page.html',
  styleUrls: ['train.page.scss']
})
export class TrainPage implements OnInit {
  SERVER_URL = 'http://localhost:5000/train';
  uploadComplete = false;
  training = false;
  results;

  constructor(
    private http: HttpClient,
    private route: ActivatedRoute
  ) {}

  ngOnInit() {
    this.uploadComplete = this.route.snapshot.params.upload;
  }

  startTraining() {
    this.training = true;

    this.http.post(this.SERVER_URL, {}).subscribe(
      (res) => {
        this.training = false;
        this.results = res;
      }
    );
  }
}
