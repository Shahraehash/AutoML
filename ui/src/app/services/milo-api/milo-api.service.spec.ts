import { TestBed } from '@angular/core/testing';

import { MiloApiService } from './milo-api.service';

describe('MiloApiService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: MiloApiService = TestBed.get(MiloApiService);
    expect(service).toBeTruthy();
  });
});
