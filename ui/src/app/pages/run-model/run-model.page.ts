import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { BackendService } from '../../services/backend.service';

@Component({
  selector: 'app-run-model',
  templateUrl: './run-model.page.html',
  styleUrls: ['./run-model.page.scss'],
})
export class RunModelPage {
  id: string;
  features: string;
  error = false;

  constructor(
    private backend: BackendService,
    private route: ActivatedRoute
  ) {
    this.route.params.subscribe(params => {
      this.id = params.id;

      this.backend.getModelFeatures(this.id).subscribe(
        features => {
          this.error = false;
          this.features = features;
        },
        () => this.error = true
      );
    });
  }
}
