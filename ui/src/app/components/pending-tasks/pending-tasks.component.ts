import { Component, Input } from '@angular/core';

import { PendingTasks } from '../../interfaces';

@Component({
  selector: 'app-pending-tasks',
  templateUrl: './pending-tasks.component.html',
  styleUrls: ['./pending-tasks.component.scss'],
})
export class PendingTasksComponent {
  @Input() pendingTasks: PendingTasks;

  constructor() {}
}
