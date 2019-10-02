import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-train',
  templateUrl: 'train.page.html',
  styleUrls: ['train.page.scss']
})
export class TrainPage implements OnInit {
  uploadComplete = false;
  training = false;
  results;

  constructor(
    private backend: BackendService,
    private route: ActivatedRoute
  ) {}

  ngOnInit() {
    this.uploadComplete = this.route.snapshot.params.upload;
  }

  startTraining() {
    this.training = true;

    this.backend.startTraining().subscribe(
      (res) => {
        this.training = false;
        this.results = res;
      }
    );
  }
}
