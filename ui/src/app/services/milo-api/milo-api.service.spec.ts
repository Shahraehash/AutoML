import { TestBed } from '@angular/core/testing';

import { MiloApiService } from './milo-api.service';

describe('MiloApiService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: MiloApiService = TestBed.inject(MiloApiService);
    expect(service).toBeTruthy();
  });
});
