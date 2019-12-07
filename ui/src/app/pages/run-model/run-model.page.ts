import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-run-model',
  templateUrl: './run-model.page.html',
  styleUrls: ['./run-model.page.scss'],
})
export class RunModelPage implements OnInit {
  id: string;

  constructor(
    private route: ActivatedRoute
  ) {
    this.route.params.subscribe( params => this.id = params.id );
  }

  ngOnInit() {}
}
