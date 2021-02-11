import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'local',
  localUser: true,
  apiUrl: 'http://localhost:5000',
};
