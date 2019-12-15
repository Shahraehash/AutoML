import { Component, HostListener } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss']
})
export class AppComponent {
  @HostListener('click', ['$event'])
  public handleExternalLinks(event) {
    if (event.target.classList.contains('external-link')) {
      event.preventDefault();
      window.open(event.target.href, '_blank');
    }
  }
}
