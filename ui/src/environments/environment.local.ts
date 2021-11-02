import { environment as defaults } from './environment.default';

export const environment = {
  ...defaults,
  name: 'local',
  localUser: true,
  apiUrl: 'http://127.0.0.1:5000',
};
