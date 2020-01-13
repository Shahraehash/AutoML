import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-feature-details',
  templateUrl: './feature-details.component.html',
  styleUrls: ['./feature-details.component.scss'],
})
export class FeatureDetailsComponent implements OnInit {
  @Input() data;

  constructor() { }

  ngOnInit() {}

}
