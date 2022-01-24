import { Component } from '@angular/core';
import { FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AlertController } from '@ionic/angular';

import { environment } from '../../../environments/environment';
import { version } from '../../../../../package.json';
import { MiloApiService } from '../../services';

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
})
export class HomePageComponent {
  version = version;
  license = new FormControl('', [Validators.required]);

  constructor(
    public api: MiloApiService,
    private router: Router,
    private alertController: AlertController
  ) {}

  get isDocker() {
    return environment.name === 'docker';
  }

  async activate() {
    try {
      await this.api.activateLicense(this.license.value.trim());
      this.router.navigate(['/']);
    } catch(err) {
      (await this.alertController.create({
        buttons: ['Dismiss'],
        message: 'Invalid license code. Please try again.'
      })).present();
    }
  }
}
