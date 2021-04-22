import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { MiloApiService } from '../../services/milo-api/milo-api.service';
import { RefitGeneralization } from '../../interfaces';

@Component({
  selector: 'app-run-model',
  templateUrl: './run-model.page.html',
  styleUrls: ['./run-model.page.scss'],
})
export class RunModelPageComponent {
  id: string;
  features: string;
  generalization: RefitGeneralization;
  threshold: number;
  error = false;

  constructor(
    private api: MiloApiService,
    private route: ActivatedRoute
  ) {
    this.route.params.subscribe(async params => {
      this.id = params.id;
      (await this.api.getModelFeatures(this.id)).subscribe(
        reply => {
          this.error = false;
          this.features = reply.features;
          this.generalization = reply.generalization;
          this.threshold = reply.threshold;
        },
        () => this.error = true
      );
    });
  }
}
