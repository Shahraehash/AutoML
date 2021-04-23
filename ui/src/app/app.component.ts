import { Component, HostListener } from '@angular/core';

import { UpdateService } from './services';

@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss']
})
export class AppComponent {
  constructor(
    public updateService: UpdateService
  ) {}

  @HostListener('click', ['$event'])
  public handleExternalLinks(event) {
    if (event.target.classList.contains('external-link')) {
      event.preventDefault();
      window.open(event.target.href, '_blank');
    }
  }
}
