import { Component, Input, OnInit } from '@angular/core';
import { Observable, timer } from 'rxjs';
import { switchMap } from 'rxjs/operators';

import { PendingTasks } from '../../interfaces';
import { BackendService } from 'src/app/services/backend.service';

@Component({
  selector: 'app-pending-tasks',
  templateUrl: './pending-tasks.component.html',
  styleUrls: ['./pending-tasks.component.scss'],
})
export class PendingTasksComponent implements OnInit {
  @Input() firstViewData: PendingTasks;
  pendingTasks$: Observable<PendingTasks>;

  constructor(
    private backend: BackendService
  ) {}

  ngOnInit() {
    this.pendingTasks$ = timer(0, 5000).pipe(
      switchMap(() => this.backend.getPendingTasks())
    );
  }
}
