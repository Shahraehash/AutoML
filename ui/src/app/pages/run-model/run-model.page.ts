import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { MiloApiService } from '../../services/milo-api/milo-api.service';

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
    private api: MiloApiService,
    private route: ActivatedRoute
  ) {
    this.route.params.subscribe(async params => {
      this.id = params.id;

      (await this.api.getModelFeatures(this.id)).subscribe(
        features => {
          this.error = false;
          this.features = features;
        },
        () => this.error = true
      );
    });
  }
}
